import Testing
import Foundation
@testable import WhizKit

// HTTPChatClient's contract with _post_chat: request shape, auth, the
// retry ladder (429/5xx/connection retried with backoff; everything else
// fails fast with the body prefix and the vision-rejection hint), and
// list_models' two shapes.
//
// .serialized because URLProtocol mocking is process-global — parallel
// tests would fight over the handler.

final class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) throws -> (Int, Data))?
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        // URLSession turns httpBody into a stream before the protocol sees
        // it, so materialize the bytes through httpBodyStream instead.
        Self.bodies.append(Self.readBody(request))
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        do {
            let (status, data) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}

    private static func readBody(_ request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 16 * 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: bufferSize)
            guard read > 0 else { break }
            data.append(buffer, count: read)
        }
        return data
    }

    static func reset() {
        handler = nil
        requests = []
        bodies = []
    }
}

@Suite(.serialized)
struct AIHTTPClientTests {

    private func client(apiKey: String = "", frames: [URL] = [], maxFrames: Int = 0)
        -> AnalysisEngine.HTTPChatClient
    {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return AnalysisEngine.HTTPChatClient(
            baseURL: "http://localhost:11434/v1",
            model: "llava",
            apiKey: apiKey,
            retryBaseDelay: 0,
            backoff: { _ in },
            session: URLSession(configuration: configuration))
    }

    private static let okBody = """
        {"choices":[{"message":{"content":"  hello from the model  "}}]}
        """

    @Test("a successful call returns trimmed content with the OpenAI request shape")
    func requestShape() async throws {
        defer { MockURLProtocol.reset() }
        MockURLProtocol.handler = { _ in (200, Data(Self.okBody.utf8)) }

        let text = try await client(apiKey: "sk-secret").complete(
            prompt: "hello", frames: [], maxFrames: 0)
        #expect(text == "hello from the model")

        let request = MockURLProtocol.requests[0]
        #expect(request.url?.absoluteString == "http://localhost:11434/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-secret")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")

        let body = (try JSONSerialization.jsonObject(
            with: MockURLProtocol.bodies[0])) as? [String: Any]
        #expect((body?["model"] as? String) == "llava")
        #expect((body?["stream"] as? Bool) == false)
        #expect((body?["temperature"] as? Double) == 0.3)
        let messages = body?["messages"] as? [[String: Any]]
        #expect((messages?[0]["role"] as? String) == "user")
        #expect((messages?[0]["content"] as? String) == "hello")
    }

    @Test("vision requests build image_url parts from frame bytes")
    func visionContent() async throws {
        defer { MockURLProtocol.reset() }
        MockURLProtocol.handler = { _ in (200, Data(Self.okBody.utf8)) }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-http-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let frame = dir.appendingPathComponent("f.jpg")
        try Data([0xFF, 0xD8, 0x00]).write(to: frame)

        _ = try await client().complete(prompt: "look", frames: [frame], maxFrames: 5)

        let body = (try JSONSerialization.jsonObject(
            with: MockURLProtocol.bodies[0])) as? [String: Any]
        let messages = body?["messages"] as? [[String: Any]]
        let parts = messages?[0]["content"] as? [[String: Any]]
        #expect(parts?.count == 2)
        #expect((parts?[0]["type"] as? String) == "text")
        #expect((parts?[1]["type"] as? String) == "image_url")
        let image = parts?[1]["image_url"] as? [String: String]
        #expect(image?["url"] == "data:image/jpeg;base64,/9gA")
    }

    @Test("a 500 retries and then succeeds")
    func retries500() async throws {
        defer { MockURLProtocol.reset() }
        var calls = 0
        MockURLProtocol.handler = { _ in
            calls += 1
            return calls < 2 ? (500, Data("boom".utf8)) : (200, Data(Self.okBody.utf8))
        }
        let text = try await client().complete(prompt: "x", frames: [], maxFrames: 0)
        #expect(text == "hello from the model")
        #expect(MockURLProtocol.requests.count == 2)
    }

    @Test("a 400 is not retried, with the vision hint when frames were sent")
    func error400NotRetried() async throws {
        defer { MockURLProtocol.reset() }
        MockURLProtocol.handler = { _ in
            (404, Data("{\"error\":\"images not supported\"}".utf8))
        }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiz-http-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let frame = dir.appendingPathComponent("f.jpg")
        try Data([0xFF]).write(to: frame)

        do {
            _ = try await client().complete(prompt: "x", frames: [frame], maxFrames: 1)
            Issue.record("expected an error")
        } catch let error as AnalysisError {
            // Not retried, and the hint points at the model, not the server.
            #expect(MockURLProtocol.requests.count == 1)
            guard case .http(let status, _, _, let hint) = error else {
                Issue.record("wrong error: \(error)")
                return
            }
            #expect(status == 404)
            #expect(hint.contains("vision-capable"))
        }
    }

    @Test("connection errors retry and end with the Ollama hint")
    func retriesConnectionErrors() async throws {
        defer { MockURLProtocol.reset() }
        MockURLProtocol.handler = { _ in
            throw URLError(.cannotConnectToHost)
        }
        do {
            _ = try await client().complete(prompt: "x", frames: [], maxFrames: 0)
            Issue.record("expected an error")
        } catch let error as AnalysisError {
            #expect(MockURLProtocol.requests.count == 3)   // 3 attempts
            guard case .unreachable(let url, _) = error else {
                Issue.record("wrong error: \(error)")
                return
            }
            #expect(url == "http://localhost:11434/v1/chat/completions")
            #expect(error.errorDescription?.contains("Is Ollama running?") == true)
        }
    }

    @Test("a response without choices raises")
    func noChoices() async throws {
        defer { MockURLProtocol.reset() }
        MockURLProtocol.handler = { _ in (200, Data("{}".utf8)) }
        do {
            _ = try await client().complete(prompt: "x", frames: [], maxFrames: 0)
            Issue.record("expected an error")
        } catch let error as AnalysisError {
            guard case .noChoices = error else {
                Issue.record("wrong error: \(error)")
                return
            }
        }
    }

    // MARK: list_models (test_ai.py:484-536)

    @Test("model listing prefers /api/tags and falls back to /v1/models")
    func listModels() async {
        defer { MockURLProtocol.reset() }
        MockURLProtocol.handler = { request in
            #expect(request.url!.path.contains("/api/tags"))
            return (200, Data(#"{"models":[{"name":"llava:13b"},{"name":"nomic"}]}"#.utf8))
        }
        let names = await AnalysisEngine.listModels(
            baseURL: "http://localhost:11434/v1",
            session: mockSession())
        #expect(names == ["llava:13b", "nomic"])

        // OpenAI shape fallback when /api/tags is missing.
        MockURLProtocol.handler = { request in
            if request.url!.path.contains("/api/tags") {
                throw URLError(.cannotConnectToHost)
            }
            return (200, Data(#"{"data":[{"id":"gpt-4o"},{"id":"m"}]}"#.utf8))
        }
        let fallback = await AnalysisEngine.listModels(
            baseURL: "http://localhost:11434/v1", session: mockSession())
        #expect(fallback == ["gpt-4o", "m"])

        // Server down → empty, never an error.
        MockURLProtocol.handler = { _ in throw URLError(.cannotConnectToHost) }
        let down = await AnalysisEngine.listModels(
            baseURL: "http://localhost:11434/v1", session: mockSession())
        #expect(down.isEmpty)
    }

    private func mockSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: configuration)
    }
}
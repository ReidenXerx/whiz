// Shared with macos/Tests/WhizLicenseTests/LicenseKeyTests.swift
// (the app verifies this key; both issuers encode the same payload). A test-only seed:
// bytes 1…32. Never a real signing key.
export const VECTOR = {
  seed: "AQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyA=",
  publicKey: "ebVWLo/mVPlAeLES6KmLp5AfhTrmlb7X4OORC60ElmQ=",
  license: { id: "txn_01testvector00000000000000", name: "Олена O'Brien / Ølsen", issued: "2026-10-21", major: 1 },
  key: "WHIZ1-eyJpZCI6InR4bl8wMXRlc3R2ZWN0b3IwMDAwMDAwMDAwMDAwMCIsImlzc3VlZCI6IjIwMjYtMTAtMjEiLCJtYWpvciI6MSwibmFtZSI6ItCe0LvQtdC90LAgTydCcmllbiAvIMOYbHNlbiIsInByb2R1Y3QiOiJ3aGl6In0.i0o0drm-XCCvUlxp191W7KwE0uuxs6DAtCipVpACyDZmgityrQZ6PdNcoJgA58OO6gZ00lInoYBoCM7YtqLiAg",
};

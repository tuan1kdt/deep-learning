import { describe, expect, it } from "vitest";
import { formatProb, stripDataUrl } from "./lib";

describe("stripDataUrl", () => {
  it("bỏ tiền tố data URL, giữ base64", () => {
    expect(stripDataUrl("data:image/png;base64,QUJD")).toBe("QUJD");
  });
  it("trả nguyên chuỗi nếu không có tiền tố", () => {
    expect(stripDataUrl("QUJD")).toBe("QUJD");
  });
});

describe("formatProb", () => {
  it("format xác suất thành phần trăm 1 chữ số", () => {
    expect(formatProb(0.8123)).toBe("81.2%");
  });
});

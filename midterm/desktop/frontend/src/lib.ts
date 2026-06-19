/** Bỏ tiền tố "data:...;base64," của FileReader, chỉ giữ phần base64 thuần
 *  để gửi sang Go (Predict nhận base64 không tiền tố). */
export function stripDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

/** 0.8123 → "81.2%" */
export function formatProb(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

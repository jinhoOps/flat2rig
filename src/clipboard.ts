import { invoke, isTauri } from "@tauri-apps/api/core";

/** Extensions the app opens from a pasted file or path (illustration or project). */
const OPENABLE = /\.(png|jpe?g|flat2rig)$/i;

/**
 * Extracts one openable file path from pasted text: a bare Windows / POSIX path, a quoted path
 * (Explorer "Copy as path"), or a `file://` URL. Returns null for anything else (multi-line text,
 * other extensions, relative paths).
 */
export function parseClipboardPath(text: string): string | null {
  let s = text.trim();
  if (!s || /[\r\n]/.test(s)) return null;
  if (s.length >= 2 && /^["']/.test(s) && s[s.length - 1] === s[0]) s = s.slice(1, -1).trim();
  if (/^file:\/\//i.test(s)) {
    try {
      const url = new URL(s);
      let p = decodeURIComponent(url.pathname);
      if (/^\/[A-Za-z]:\//.test(p)) p = p.slice(1).replace(/\//g, "\\");
      else if (url.host) p = `\\\\${url.host}${p.replace(/\//g, "\\")}`;
      s = p;
    } catch {
      return null;
    }
  }
  const absolute = /^[A-Za-z]:[\\/]/.test(s) || /^\\\\[^\\]/.test(s) || s.startsWith("/");
  if (!absolute || !OPENABLE.test(s)) return null;
  return s;
}

/** First pasted image or openable file (screenshot bitmap, or a file copied in Explorer). */
export function clipboardFile(data: DataTransfer | null): File | null {
  if (!data) return null;
  const files = Array.from(data.files ?? []);
  const named = files.find((f) => OPENABLE.test(f.name));
  if (named) return named;
  const image = files.find((f) => /^image\/(png|jpeg)$/.test(f.type));
  if (image) return withImageName(image);
  for (const item of Array.from(data.items ?? [])) {
    if (item.kind === "file" && /^image\/(png|jpeg)$/.test(item.type)) {
      const f = item.getAsFile();
      if (f) return withImageName(f);
    }
  }
  return null;
}

function withImageName(f: File): File {
  if (OPENABLE.test(f.name)) return f;
  const ext = f.type === "image/jpeg" ? "jpg" : "png";
  return new File([f], `clipboard.${ext}`, { type: f.type });
}

/** Reads a pasted path through the desktop backend (the webview cannot open arbitrary paths). */
export async function readClipboardPath(path: string): Promise<File> {
  if (!isTauri())
    throw new Error("경로 붙여넣기는 데스크톱 앱에서만 됩니다. 이미지를 복사해 붙여넣거나 파일을 끌어 놓으세요.");
  const res = await invoke<{ name: string; bytes: string }>("read_pasted_path", { path });
  const bin = atob(res.bytes);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const type = /\.png$/i.test(res.name)
    ? "image/png"
    : /\.jpe?g$/i.test(res.name)
      ? "image/jpeg"
      : "application/octet-stream";
  return new File([bytes], res.name, { type });
}

/** Whether a paste event targets a text field (then the app leaves it alone). */
export function isTextEditingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLInputElement)
    return !["button", "checkbox", "radio", "range", "file", "color", "submit", "reset"].includes(
      target.type,
    );
  return false;
}

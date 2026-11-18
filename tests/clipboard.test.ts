import { describe, expect, it } from "vitest";
import { clipboardFile, parseClipboardPath } from "../src/clipboard";

describe("clipboard paste input", () => {
  it("accepts absolute image / project paths in the forms Windows copies them", () => {
    expect(parseClipboardPath("C:\\Users\\kjh\\orca\\flat2rig\\ref\\sutteang\\2d_.png")).toBe(
      "C:\\Users\\kjh\\orca\\flat2rig\\ref\\sutteang\\2d_.png",
    );
    // Explorer "Copy as path" quotes the path.
    expect(parseClipboardPath('  "D:\\art\\char 01.JPG"\r\n'.trimEnd())).toBe("D:\\art\\char 01.JPG");
    expect(parseClipboardPath("file:///C:/art/%ED%9E%88%EB%AF%B8.png")).toBe("C:\\art\\히미.png");
    expect(parseClipboardPath("\\\\nas\\share\\proj.flat2rig")).toBe("\\\\nas\\share\\proj.flat2rig");
    expect(parseClipboardPath("/home/me/art.jpeg")).toBe("/home/me/art.jpeg");
  });

  it("ignores text that is not one openable absolute path", () => {
    expect(parseClipboardPath("")).toBeNull();
    expect(parseClipboardPath("hello")).toBeNull();
    expect(parseClipboardPath("ref/sutteang/2d_.png")).toBeNull();
    expect(parseClipboardPath("C:\\art\\model.psd")).toBeNull();
    expect(parseClipboardPath("C:\\a.png\nC:\\b.png")).toBeNull();
    expect(parseClipboardPath("https://example.com/a.png")).toBeNull();
  });

  it("names a pasted bitmap so the image loader accepts it", () => {
    const bitmap = new File([new Uint8Array([1])], "", { type: "image/png" });
    const data = { files: [bitmap], items: [] } as unknown as DataTransfer;
    const f = clipboardFile(data);
    expect(f?.name).toBe("clipboard.png");
    expect(f?.type).toBe("image/png");
    const project = new File([new Uint8Array([1])], "scene.flat2rig");
    expect(clipboardFile({ files: [project], items: [] } as unknown as DataTransfer)?.name).toBe(
      "scene.flat2rig",
    );
    const text = new File([new Uint8Array([1])], "a.txt", { type: "text/plain" });
    expect(clipboardFile({ files: [text], items: [] } as unknown as DataTransfer)).toBeNull();
  });
});

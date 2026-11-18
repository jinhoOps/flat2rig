import { useEffect, useRef, useState, type ReactNode } from "react";
import { Check } from "lucide-react";

export type MenuItem =
  | {
      label: string;
      shortcut?: string;
      disabled?: boolean;
      checked?: boolean;
      onSelect: () => void;
    }
  | "separator";

export interface MenuDef {
  id: string;
  label: string;
  items: MenuItem[];
}

/**
 * Top menu row (파일 · 편집 · 보기 · 도움말). Every entry also exists as a direct control
 * somewhere in the workspace; the menus only collect them. Save / export stay as buttons on the
 * right of the title row so they are always one click away.
 */
export default function MenuBar({
  menus,
  disabled,
  children,
}: {
  menus: MenuDef[];
  disabled?: boolean;
  children?: ReactNode;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(null);
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", esc);
    };
  }, [open]);
  return (
    <div className="ws-menus" ref={root} role="menubar" aria-label="메뉴">
      {children}
      {menus.map((m) => (
        <div key={m.id} className="mb-menu">
          <button
            type="button"
            role="menuitem"
            aria-haspopup="menu"
            aria-expanded={open === m.id}
            className={open === m.id ? "is-open" : ""}
            disabled={disabled}
            onClick={() => setOpen((v) => (v === m.id ? null : m.id))}
            onMouseEnter={() => {
              if (open && open !== m.id) setOpen(m.id);
            }}
          >
            {m.label}
          </button>
          {open === m.id && (
            <div className="mb-pop" role="menu" aria-label={m.label}>
              {m.items.map((it, i) =>
                it === "separator" ? (
                  <hr key={`sep-${i}`} />
                ) : (
                  <button
                    type="button"
                    key={it.label}
                    role={it.checked === undefined ? "menuitem" : "menuitemcheckbox"}
                    aria-checked={it.checked}
                    disabled={it.disabled}
                    onClick={() => {
                      setOpen(null);
                      it.onSelect();
                    }}
                  >
                    <span className="mb-check">{it.checked ? <Check size={12} /> : null}</span>
                    <span className="mb-label">{it.label}</span>
                    {it.shortcut && <kbd>{it.shortcut}</kbd>}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

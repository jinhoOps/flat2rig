import { useId } from "react";

/** The same two planes and pivot are used by the mark, app tile and motion. */
export function RigGeometry({ mesh = false }: { mesh?: boolean }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg className="rig-geometry" viewBox="0 0 80 80" fill="none" aria-hidden="true">
      <defs>
        <mask id={`${id}-seam`} maskUnits="userSpaceOnUse" x="-20" y="-20" width="120" height="120">
          <rect x="-20" y="-20" width="120" height="120" fill="white" />
          <g className="rig-plane">
            <rect x="32" y="16" width="27" height="46" rx="5" fill="black" stroke="black" strokeWidth="4" />
          </g>
        </mask>
        <clipPath id={`${id}-clip`}>
          <rect x="32" y="16" width="27" height="46" rx="5" />
        </clipPath>
      </defs>
      <rect className="rig-fixed" x="14" y="16" width="27" height="46" rx="5" fill="currentColor" mask={`url(#${id}-seam)`} />
      <g className="rig-plane">
        <rect x="32" y="16" width="27" height="46" rx="5" fill="#2456d6" />
        <g className="rig-mesh" data-on={mesh || undefined} clipPath={`url(#${id}-clip)`}>
          <path d="M32 16 45.5 31 59 16M32 39l13.5-8L59 39 38 53 32 39Zm13.5-8L38 53l21 9M32 62l6-9M59 39v23M32 16v46M45.5 31V16" stroke="#fff" strokeWidth=".65" strokeLinejoin="round" />
          <circle cx="45.5" cy="31" r="1.3" fill="#fff" />
          <circle cx="59" cy="39" r="1.3" fill="#fff" />
        </g>
        <circle cx="38" cy="53" r="4.2" fill="white" />
        <circle cx="38" cy="53" r="1.65" fill="#161920" />
      </g>
    </svg>
  );
}

export function BrandWordmark() {
  return <span className="brand-wordmark">flat<span className="brand-two">2</span>rig</span>;
}

export function BrandMark() {
  return <span className="brand-mark"><RigGeometry /></span>;
}

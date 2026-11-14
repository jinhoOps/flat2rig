/** A simple head and shoulders with a small triangular rigging mesh. */
export function RigGeometry({ mesh = true }: { mesh?: boolean }) {
  return (
    <svg
      className="rig-geometry"
      viewBox="0 0 80 80"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M16 68c1-9 8-14 17-16m14 0c9 2 16 7 17 16"
        stroke="currentColor"
        strokeWidth="2.6"
        strokeLinecap="round"
      />
      <g className="rig-head">
        <path
          d="M40 12c-12 0-19 9-19 21 0 14 9 25 19 25s19-11 19-25c0-12-7-21-19-21Z"
          fill="#f6f7fa"
          stroke="currentColor"
          strokeWidth="2.6"
          strokeLinejoin="round"
        />
        <g className="rig-mesh" data-on={mesh || undefined}>
          <path
            d="m40 21-12 14 12 16 12-16-12-14v30M28 35h24"
            stroke="#2456d6"
            strokeWidth="1.3"
            strokeLinejoin="round"
          />
          <g fill="#2456d6">
            <circle cx="40" cy="21" r="2" />
            <circle cx="28" cy="35" r="2" />
            <circle cx="52" cy="35" r="2" />
            <circle cx="40" cy="51" r="2" />
          </g>
        </g>
      </g>
    </svg>
  );
}

export function BrandWordmark() {
  return (
    <span className="brand-wordmark">
      flat<span className="brand-two">2</span>rig
    </span>
  );
}

export function BrandMark() {
  return (
    <span className="brand-mark">
      <RigGeometry />
    </span>
  );
}

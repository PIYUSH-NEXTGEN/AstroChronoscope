import React from "react";
import { cn } from "@/lib/utils";

function TwinOrbit({ className, style, ...props }: React.ComponentProps<"span">) {
  return (
    <>
      <style>{`
        @keyframes loading-ui-twin-orbit-spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
      <span
        role="status"
        className={cn(
          "relative inline-flex items-center justify-center shrink-0 aspect-square",
          className,
        )}
        style={style}
        {...props}
      >
        {/* Central Nucleus */}
        <span
          aria-hidden="true"
          className="size-[38%] rounded-full bg-current"
        />

        {/* Orbiting Satellite 1 */}
        <span
          aria-hidden="true"
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
          style={{
            animation: "loading-ui-twin-orbit-spin var(--duration, 1.2s) linear infinite",
          }}
        >
          <span className="absolute top-0 size-[26%] rounded-full bg-current" />
        </span>

        {/* Orbiting Satellite 2 (180 deg offset) */}
        <span
          aria-hidden="true"
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
          style={{
            animation: "loading-ui-twin-orbit-spin var(--duration, 1.2s) linear infinite",
            animationDelay: "calc(var(--duration, 1.2s) / -2)",
          }}
        >
          <span className="absolute top-0 size-[26%] rounded-full bg-current" />
        </span>

        <span className="sr-only">Loading</span>
      </span>
    </>
  );
}

export { TwinOrbit };

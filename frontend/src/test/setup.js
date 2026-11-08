// Test-only shims: jsdom has no layout engine, so give charts a fixed size and stub ResizeObserver.
import React from "react";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(cleanup);

globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };

vi.mock("recharts", async (importOriginal) => {
  const mod = await importOriginal();
  const ResponsiveContainer = ({ children, height }) =>
    React.cloneElement(children, { width: 640, height: typeof height === "number" ? height : 300 });
  return { ...mod, ResponsiveContainer };
});

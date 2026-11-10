import katex from 'katex';
import 'katex/dist/katex.min.css';
import { Fragment, useEffect, useRef } from 'react';
import type { ReactNode } from 'react';

/**
 * Text with formulas and citation markers.
 *
 * Models write maths as LaTeX: $x^2$ and \(x^2\) inline, $$...$$ and \[...\]
 * displayed. Shown as plain text that reads as noise, so those spans are
 * typeset with KaTeX. [n] markers outside a formula become buttons that open
 * the cited chunk.
 *
 * Plain-text maths with no delimiters (f'(g(x)) in the source notes) is left
 * alone: there is no reliable way to tell it from prose.
 */

// $$..$$, \[..\], \(..\), then $..$. The single-dollar form must not start or
// end with a space and cannot span lines, so "costs $5 and $6" stays prose.
const MATH =
  /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?![\s$])([^$\n]+?)(?<![\s$])\$/g;

function Tex({ source, display }: { source: string; display: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    // throwOnError: false shows bad LaTeX as red source text instead of
    // breaking the page; trust: false refuses \href, \url and friends.
    katex.render(source.trim(), ref.current, {
      displayMode: display,
      throwOnError: false,
      trust: false,
      strict: 'ignore',
      output: 'htmlAndMathml',
    });
  }, [source, display]);
  return <span ref={ref} className={display ? 'math-block' : 'math'} />;
}

function withCitations(text: string, onOpen?: ((n: number) => void) | undefined): ReactNode[] {
  return text.split(/(\[\d+\])/g).map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match || !onOpen) return <Fragment key={index}>{part}</Fragment>;
    return (
      <button key={index} type="button" className="cite" onClick={() => onOpen(Number(match[1]))}>
        {part}
      </button>
    );
  });
}

export function RichText({
  text,
  onCite,
}: {
  text: string;
  onCite?: ((n: number) => void) | undefined;
}): ReactNode {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(MATH)) {
    const start = m.index ?? 0;
    const display = m[1] !== undefined || m[2] !== undefined;
    let before = text.slice(last, start);
    // A displayed formula is its own block with its own margin, so the blank
    // lines the model puts around it would double the gap.
    if (display) before = before.replace(/\s+$/, '');
    if (before) out.push(...withCitations(before, onCite));
    out.push(
      <Tex key={`m${start}`} source={m[1] ?? m[2] ?? m[3] ?? m[4] ?? ''} display={display} />,
    );
    last = start + m[0].length;
    if (display) last += /^\s*/.exec(text.slice(last))?.[0].length ?? 0;
  }
  if (last < text.length) out.push(...withCitations(text.slice(last), onCite));
  return <>{out}</>;
}

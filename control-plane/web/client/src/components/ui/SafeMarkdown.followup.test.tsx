import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { expectSafeMarkdown } from "@/test/markdownSecurity";
import { SafeMarkdown } from "./SafeMarkdown";

afterEach(cleanup);

describe("SafeMarkdown review follow-ups", () => {
  it.each([
    ["soft", "first\nsecond", "first<br>\nsecond"],
    ["hard", "**first**  \nsecond", "<strong>first</strong><br>\nsecond"],
  ])("C4: a paragraph %s break produces exactly one structural line break", (_name, content, html) => {
    const { container } = render(<SafeMarkdown content={content} />);
    const paragraph = container.querySelector("p")!;
    expect(paragraph.innerHTML).toBe(html);
    expect(paragraph.querySelectorAll("br")).toHaveLength(1);
    expect(paragraph).not.toHaveClass("whitespace-pre-line");
  });

  it("C4: a tight list continuation has one line break inside the list item", () => {
    const { container } = render(<SafeMarkdown content={"- **first**\n  continuation"} />);
    const item = screen.getByRole("listitem");
    expect(item.innerHTML).toBe("<strong>first</strong><br>\ncontinuation");
    expect(container.querySelectorAll("br")).toHaveLength(1);
    expect(item.querySelector("p")).toBeNull();
  });

  it.each([
    ["fenced", "```text\nfirst\nsecond\n```"],
    ["indented", "    first\n    second"],
  ])("C4: %s code keeps its newlines without br elements", (_name, content) => {
    const { container } = render(<SafeMarkdown content={content} />);
    expect(container.querySelector("pre code")?.textContent).toBe("first\nsecond\n");
    expect(container.querySelector("br")).toBeNull();
  });

  it("C4: inline code retains normal markdown whitespace and literal content", () => {
    const { container } = render(<SafeMarkdown content={"before `first\n<img>` after"} />);
    expect(container.querySelector("p")?.innerHTML).toBe(
      'before <code class="bg-muted px-1 rounded text-sm font-mono">first &lt;img&gt;</code> after',
    );
    expect(container.querySelector("br")).toBeNull();
    expectSafeMarkdown(container);
  });

  it("C4: ordered lists retain their numeric start", () => {
    render(<SafeMarkdown content={"5. Fifth\n6. Sixth"} />);
    expect(screen.getByRole("list")).toHaveAttribute("start", "5");
    expect(screen.getAllByRole("listitem").map((item) => item.textContent)).toEqual(["Fifth", "Sixth"]);
  });

  it("C2: a javascript reference definition leaves visible text without a link", () => {
    const { container } = render(<SafeMarkdown content={"[x][r]\n\n[r]: javascript:alert(1)"} />);
    expect(container.querySelector("p")?.innerHTML).toBe("x");
    expect(screen.queryByRole("link")).toBeNull();
    expectSafeMarkdown(container);
  });

  it("C3: a reference-style image renders only its alt text", () => {
    const { container } = render(<SafeMarkdown content={"![reference alt][r]\n\n[r]: https://example.com/image.png"} />);
    expect(container.querySelector("p")?.innerHTML).toBe("reference alt");
    expectSafeMarkdown(container);
  });

  it("C1: fenced code info markup stays inside the class attribute", () => {
    const { container } = render(<SafeMarkdown content={'```"><img src=x>\ncode\n```'} />);
    const code = container.querySelector("pre code")!;
    expect(code.textContent).toBe("code\n");
    expect(code.getAttributeNames()).toEqual(["class"]);
    expect(code).toHaveClass('language-"><img');
    expectSafeMarkdown(container);
  });

  it("C1 C2: GFM footnotes retain their text without unsafe fragment links", () => {
    const { container } = render(<SafeMarkdown content={"A note[^r].\n\n[^r]: Footnote **text**."} />);
    expect(container.querySelector("p")?.textContent).toBe("A note1.");
    expect(container.textContent).toContain("Footnote text.");
    expect(screen.queryByRole("link")).toBeNull();
    expectSafeMarkdown(container);
  });

  it.each([
    '<base href="https://evil.example/">',
    '<math href="javascript:alert(1)">text</math>',
    '<meta http-equiv="refresh" content="0;url=https://evil.example/">',
    '<link rel="stylesheet" href="https://evil.example/style.css">',
  ])("C1: raw forbidden element remains literal text: %s", (content) => {
    const { container } = render(<SafeMarkdown content={content} />);
    expect(container.textContent).toBe(content);
    expectSafeMarkdown(container);
  });
});

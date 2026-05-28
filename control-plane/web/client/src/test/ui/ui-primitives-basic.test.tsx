// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { Badge, StatusBadge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  cardVariants,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    X: Icon,
    Check: Icon,
    ChevronDown: Icon,
    ChevronUp: Icon,
  };
});

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => {};
  Element.prototype.releasePointerCapture ??= () => {};
  Element.prototype.scrollIntoView ??= () => {};
});

describe("ui primitive components", () => {
  it("renders button variants and supports asChild links", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();

    render(
      <div>
        <Button variant="destructive" size="lg" onClick={onClick}>
          Delete
        </Button>
        <Button asChild variant="link">
          <a href="/docs">Docs</a>
        </Button>
      </div>,
    );

    expect(buttonVariants({ variant: "destructive", size: "lg" })).toContain("bg-destructive");
    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(onClick).toHaveBeenCalledTimes(1);

    const link = screen.getByRole("link", { name: "Docs" });
    expect(link).toHaveAttribute("href", "/docs");
    expect(link.className).toContain("underline-offset-4");
  });

  it("renders inputs and card subcomponents", () => {
    const { container } = render(
      <div>
        <Input type="email" placeholder="name@example.com" disabled />
        <Card variant="outline" interactive={false}>
          <CardHeader>
            <CardTitle>Header</CardTitle>
            <CardDescription>Description</CardDescription>
          </CardHeader>
          <CardContent>Body</CardContent>
          <CardFooter>Footer</CardFooter>
        </Card>
      </div>,
    );

    expect(screen.getByPlaceholderText("name@example.com")).toBeDisabled();
    expect(cardVariants({ variant: "outline", interactive: false })).toContain("shadow-none");
    expect(screen.getByText("Header").tagName).toBe("H3");
    expect(screen.getByText("Body")).toBeInTheDocument();
    expect(container.querySelector(".p-6.pt-0")).toBeTruthy();
  });

  it("renders badges for variants and status icons", () => {
    render(
      <div>
        <Badge variant="metadata">meta</Badge>
        <Badge variant="running">running</Badge>
        <StatusBadge status="failed" />
      </div>,
    );

    expect(screen.getByText("meta").className).toContain("font-mono");
    expect(screen.getByText("running").previousSibling).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
  });

  it("opens and closes a dialog via trigger and close button", async () => {
    const user = userEvent.setup();

    render(
      <Dialog>
        <DialogTrigger asChild>
          <Button>Open dialog</Button>
        </DialogTrigger>
        <DialogContent>
          <DialogTitle>Dialog title</DialogTitle>
          <DialogDescription>Dialog body</DialogDescription>
        </DialogContent>
      </Dialog>,
    );

    await user.click(screen.getByRole("button", { name: "Open dialog" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Dialog body")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("switches tabs and renders table sections", async () => {
    const user = userEvent.setup();

    render(
      <div>
        <Tabs defaultValue="two">
          <TabsList variant="underline" density="relaxed">
            <TabsTrigger value="one" variant="underline">
              One
            </TabsTrigger>
            <TabsTrigger value="two" variant="underline" size="lg">
              Two
            </TabsTrigger>
          </TabsList>
          <TabsContent value="one">First panel</TabsContent>
          <TabsContent value="two">Second panel</TabsContent>
        </Tabs>

        <Table>
          <TableCaption>Example table</TableCaption>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow data-state="selected">
              <TableCell>Row one</TableCell>
            </TableRow>
          </TableBody>
          <TableFooter>
            <TableRow>
              <TableCell>Total</TableCell>
            </TableRow>
          </TableFooter>
        </Table>
      </div>,
    );

    expect(screen.getByText("Second panel")).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "One" }));
    expect(screen.getByText("First panel")).toBeVisible();
    expect(screen.getByText("Example table")).toBeInTheDocument();
    expect(screen.getByText("Row one").closest("tr")?.className).toContain("data-[state=selected]:bg-muted");
  });

  it("toggles switch state and shows tooltip, popover, and select content", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    const onValueChange = vi.fn();

    const { unmount } = render(
      <div>
        <Switch aria-label="Enabled" onCheckedChange={onCheckedChange} />

        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <TooltipTrigger asChild>
              <button type="button">Hover me</button>
            </TooltipTrigger>
            <TooltipContent>Helpful text</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <Popover>
          <PopoverTrigger asChild>
            <button type="button">Open popover</button>
          </PopoverTrigger>
          <PopoverContent>Popover body</PopoverContent>
        </Popover>
      </div>,
    );

    await user.click(screen.getByRole("switch", { name: "Enabled" }));
    expect(onCheckedChange).toHaveBeenCalledWith(true);

    await user.hover(screen.getByRole("button", { name: "Hover me" }));
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Helpful text");

    await user.click(screen.getByRole("button", { name: "Open popover" }));
    expect(await screen.findByText("Popover body")).toBeInTheDocument();

    unmount();

    render(
      <Select open onValueChange={onValueChange}>
        <SelectTrigger aria-label="Fruit">
          <SelectValue placeholder="Pick one" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="apple">Apple</SelectItem>
          <SelectItem value="pear">Pear</SelectItem>
        </SelectContent>
      </Select>,
    );

    const pearOption = screen.getByRole("option", { name: "Pear" });
    expect(pearOption).toBeInTheDocument();
    fireEvent.click(pearOption);
    expect(onValueChange).toHaveBeenCalledWith("pear");
  });
});
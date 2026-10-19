import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { VCChainVerifiabilityWarning } from "@/components/vc/VCChainVerifiabilityWarning";

describe("VCChainVerifiabilityWarning", () => {
  it("shows a warning with the unresolved DIDs when the chain is not verifiable", () => {
    render(
      <VCChainVerifiabilityWarning
        chain={{
          verifiable: false,
          unresolved_dids: ["did:web:example.com:agents:gone", "did:caller:client"],
        }}
      />,
    );

    expect(
      screen.getByText("This audit chain cannot be fully verified: 2 DIDs could not be resolved"),
    ).toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Unresolved DIDs"));
    expect(screen.getByText("did:web:example.com:agents:gone")).toBeInTheDocument();
    expect(screen.getByText("did:caller:client")).toBeInTheDocument();
  });

  it("truncates long DID lists", () => {
    const dids = Array.from({ length: 12 }, (_, i) => `did:web:example.com:agents:a${i}`);
    render(<VCChainVerifiabilityWarning chain={{ verifiable: false, unresolved_dids: dids }} />);

    expect(
      screen.getByText("This audit chain cannot be fully verified: 12 DIDs could not be resolved"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByText("Unresolved DIDs"));
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  it("renders nothing for a verifiable chain", () => {
    const { container } = render(
      <VCChainVerifiabilityWarning chain={{ verifiable: true, unresolved_dids: [] }} />,
    );
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText(/cannot be fully verified/)).not.toBeInTheDocument();
  });

  it("renders nothing when the server does not report verifiability", () => {
    const { container } = render(<VCChainVerifiabilityWarning chain={{}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

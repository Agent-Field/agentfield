/**
 * Agent Color Manager - Generates consistent, accessible colors for agent differentiation
 * Uses HSL color space with golden ratio distribution for optimal visual separation
 */

interface AgentColor {
  primary: string; // Full opacity color for badges/accents
  background: string; // Low opacity for backgrounds
  border: string; // Medium opacity for borders
  text: string; // High contrast text color
  name: string; // Agent name for reference
}

interface ColorCache {
  [agentKey: string]: AgentColor;
}

class AgentColorManager {
  private colorCache: ColorCache = {};
  private usedHues: number[] = [];
  private readonly goldenRatio = 0.618033988749;

  // Curated color palette with good visual separation
  private readonly predefinedColors = [
    { h: 210, s: 70, l: 55 }, // Blue
    { h: 280, s: 65, l: 60 }, // Purple
    { h: 340, s: 70, l: 55 }, // Pink
    { h: 25, s: 75, l: 55 }, // Orange
    { h: 180, s: 60, l: 50 }, // Cyan
    { h: 260, s: 60, l: 65 }, // Violet
    { h: 45, s: 70, l: 55 }, // Amber
    { h: 160, s: 65, l: 50 }, // Teal
    { h: 300, s: 65, l: 60 }, // Magenta
    { h: 200, s: 60, l: 55 }, // Light Blue
    { h: 320, s: 70, l: 60 }, // Rose
    { h: 60, s: 65, l: 55 }, // Yellow-Green
  ];

  // Colors to avoid (too similar to status indicators)
  private readonly avoidHues = [
    { min: 110, max: 140 }, // Green (success)
    { min: 0, max: 15 }, // Red (error)
    { min: 345, max: 360 }, // Red (error)
    { min: 220, max: 250 }, // Blue (running)
  ];

  /**
   * Get or generate a color for an agent
   */
  getAgentColor(agentName: string, agentId?: string): AgentColor {
    const agentKey = this.getAgentKey(agentName, agentId);

    if (this.colorCache[agentKey]) {
      return this.colorCache[agentKey];
    }

    const color = this.generateAgentColor(agentKey, agentName);
    this.colorCache[agentKey] = color;
    return color;
  }

  /**
   * Get all cached agent colors for legend display
   */
  getAllAgentColors(): AgentColor[] {
    return Object.values(this.colorCache);
  }

  /**
   * Clear color cache (useful for testing or reset)
   */
  clearCache(): void {
    this.colorCache = {};
    this.usedHues = [];
  }

  /**
   * Remove agents that are not in the provided list (workflow-specific cleanup)
   */
  cleanupUnusedAgents(activeAgentNames: string[]): void {
    const activeAgentKeys = new Set(
      activeAgentNames.map(name => this.getAgentKey(name))
    );

    // Remove agents that are no longer active
    Object.keys(this.colorCache).forEach(agentKey => {
      if (!activeAgentKeys.has(agentKey)) {
        delete this.colorCache[agentKey];
      }
    });

    // Rebuild used hues array from remaining agents
    this.usedHues = Object.values(this.colorCache).map(color => {
      // Extract hue from HSL color string
      const hslMatch = color.primary.match(/hsl\((\d+),/);
      return hslMatch ? parseInt(hslMatch[1]) : 0;
    });
  }

  /**
   * Generate a unique key for agent identification
   */
  private getAgentKey(agentName: string, agentId?: string): string {
    // Use name if meaningful, fallback to ID
    const identifier =
      agentName && agentName.length > 3 && !agentName.match(/^agent[_-]?\d+$/i)
        ? agentName
        : agentId || agentName;

    return identifier.toLowerCase().trim();
  }

  /**
   * Generate a color for an agent using multiple strategies
   */
  private generateAgentColor(agentKey: string, agentName: string): AgentColor {
    let hsl: { h: number; s: number; l: number };

    // Strategy 1: Use predefined colors for first agents
    if (Object.keys(this.colorCache).length < this.predefinedColors.length) {
      hsl = this.predefinedColors[Object.keys(this.colorCache).length];
    } else {
      // Strategy 2: Generate using hash + golden ratio
      hsl = this.generateHashBasedColor(agentKey);
    }

    // Ensure minimum distance from existing colors
    hsl = this.ensureColorSeparation(hsl);

    this.usedHues.push(hsl.h);

    return this.createColorVariants(hsl, agentName);
  }

  /**
   * Generate color based on string hash and golden ratio
   */
  private generateHashBasedColor(input: string): {
    h: number;
    s: number;
    l: number;
  } {
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    // Use golden ratio for hue distribution
    const hue = (Math.abs(hash) * this.goldenRatio * 360) % 360;

    return {
      h: Math.round(hue),
      s: 65 + (Math.abs(hash) % 15), // 65-80% saturation
      l: 50 + (Math.abs(hash) % 15), // 50-65% lightness
    };
  }

  /**
   * Ensure color is sufficiently different from existing colors and avoid status colors
   */
  private ensureColorSeparation(hsl: { h: number; s: number; l: number }): {
    h: number;
    s: number;
    l: number;
  } {
    const minHueDistance = 30; // Minimum degrees between hues
    let adjustedHue = hsl.h;

    // Check against avoided hues (status colors)
    for (const avoid of this.avoidHues) {
      if (adjustedHue >= avoid.min && adjustedHue <= avoid.max) {
        adjustedHue = avoid.max + 15; // Move away from avoided range
        break;
      }
    }

    // Check against existing hues
    let attempts = 0;
    while (attempts < 12) {
      // Max 12 attempts to find good hue
      const tooClose = this.usedHues.some((usedHue) => {
        const distance = Math.min(
          Math.abs(adjustedHue - usedHue),
          360 - Math.abs(adjustedHue - usedHue)
        );
        return distance < minHueDistance;
      });

      if (!tooClose) break;

      adjustedHue = (adjustedHue + 30) % 360; // Try next hue
      attempts++;
    }

    return { ...hsl, h: adjustedHue };
  }

  /**
   * Create color variants for different use cases
   */
  private createColorVariants(
    hsl: { h: number; s: number; l: number },
    agentName: string
  ): AgentColor {
    const { h, s, l } = hsl;

    return {
      primary: `hsl(${h}, ${s}%, ${l}%)`,
      background: `hsla(${h}, ${Math.max(s - 10, 40)}%, ${Math.min(
        l + 10,
        70
      )}%, 0.08)`,
      border: `hsla(${h}, ${s}%, ${l}%, 0.3)`,
      text: l > 60 ? `hsl(${h}, ${s}%, 20%)` : `hsl(${h}, ${s}%, 90%)`,
      name: agentName,
    };
  }

  /**
   * Get agent initials for badge display
   */
  getAgentInitials(agentName: string): string {
    if (!agentName) return "?";

    // Handle camelCase and PascalCase
    const camelCaseMatch = agentName.match(/[A-Z]/g);
    if (camelCaseMatch && camelCaseMatch.length >= 2) {
      return camelCaseMatch.slice(0, 2).join("");
    }

    // Handle space/underscore/dash separated
    const words = agentName.split(/[\s_-]+/).filter((word) => word.length > 0);
    if (words.length >= 2) {
      return words
        .slice(0, 2)
        .map((word) => word[0].toUpperCase())
        .join("");
    }

    // Single word - take first two characters
    if (agentName.length >= 2) {
      return agentName.substring(0, 2).toUpperCase();
    }

    return agentName.toUpperCase();
  }

  /**
   * Check if a color meets accessibility contrast requirements
   */
}

// Export singleton instance
export const agentColorManager = new AgentColorManager();

// Export types for use in components
export type { AgentColor };

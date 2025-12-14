package cli

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/templates"
)

// TestUpdate_TextInput_AcceptsJK verifies that j and k can be typed during text input steps
func TestUpdate_TextInput_AcceptsJK(t *testing.T) {
	tests := []struct {
		name     string
		step     int
		input    rune
		expected string
	}{
		{"step 0 accepts k", 0, 'k', "k"},
		{"step 0 accepts j", 0, 'j', "j"},
		{"step 2 accepts k", 2, 'k', "k"},
		{"step 2 accepts j", 2, 'j', "j"},
		{"step 3 accepts k", 3, 'k', "k"},
		{"step 3 accepts j", 3, 'j', "j"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := initModel{
				step:      tt.step,
				textInput: "",
				choices:   templates.GetSupportedLanguages(),
			}

			msg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{tt.input}}
			newModel, _ := model.Update(msg)
			m := newModel.(initModel)

			require.Equal(t, tt.expected, m.textInput, "character should be added to text input")
		})
	}
}

// TestUpdate_NavigationKeys_OnlyInStep1 verifies navigation keys only work in step 1
func TestUpdate_NavigationKeys_OnlyInStep1(t *testing.T) {
	// Test that k navigates up in step 1
	t.Run("k navigates up in step 1", func(t *testing.T) {
		model := initModel{
			step:    1,
			cursor:  1,
			choices: templates.GetSupportedLanguages(),
		}

		msg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'k'}}
		newModel, _ := model.Update(msg)
		m := newModel.(initModel)

		require.Equal(t, 0, m.cursor, "cursor should move up")
	})

	// Test that j navigates down in step 1
	t.Run("j navigates down in step 1", func(t *testing.T) {
		model := initModel{
			step:    1,
			cursor:  0,
			choices: templates.GetSupportedLanguages(),
		}

		msg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'j'}}
		newModel, _ := model.Update(msg)
		m := newModel.(initModel)

		require.Equal(t, 1, m.cursor, "cursor should move down")
	})

	// Test that up arrow navigates in step 1
	t.Run("up arrow navigates in step 1", func(t *testing.T) {
		model := initModel{
			step:    1,
			cursor:  1,
			choices: templates.GetSupportedLanguages(),
		}

		msg := tea.KeyMsg{Type: tea.KeyUp}
		newModel, _ := model.Update(msg)
		m := newModel.(initModel)

		require.Equal(t, 0, m.cursor, "cursor should move up")
	})

	// Test cursor wraps around
	t.Run("cursor wraps around at top", func(t *testing.T) {
		model := initModel{
			step:    1,
			cursor:  0,
			choices: templates.GetSupportedLanguages(),
		}

		msg := tea.KeyMsg{Type: tea.KeyUp}
		newModel, _ := model.Update(msg)
		m := newModel.(initModel)

		require.Equal(t, len(model.choices)-1, m.cursor, "cursor should wrap to bottom")
	})

	t.Run("cursor wraps around at bottom", func(t *testing.T) {
		model := initModel{
			step:    1,
			cursor:  len(templates.GetSupportedLanguages()) - 1,
			choices: templates.GetSupportedLanguages(),
		}

		msg := tea.KeyMsg{Type: tea.KeyDown}
		newModel, _ := model.Update(msg)
		m := newModel.(initModel)

		require.Equal(t, 0, m.cursor, "cursor should wrap to top")
	})
}

// TestUpdate_CtrlC_SetsCancelled verifies ctrl+c sets the cancelled flag
func TestUpdate_CtrlC_SetsCancelled(t *testing.T) {
	model := initModel{
		step:      0,
		textInput: "test",
		choices:   templates.GetSupportedLanguages(),
	}

	msg := tea.KeyMsg{Type: tea.KeyCtrlC}
	newModel, cmd := model.Update(msg)
	m := newModel.(initModel)

	require.True(t, m.cancelled, "cancelled should be true")
	require.True(t, m.done, "done should be true")
	require.NotNil(t, cmd, "should return quit command")
}

// TestUpdate_SpecialKeys_NotTypedAsText verifies special keys are not typed as text
func TestUpdate_SpecialKeys_NotTypedAsText(t *testing.T) {
	tests := []struct {
		name    string
		keyType tea.KeyType
	}{
		{"escape", tea.KeyEscape},
		{"tab", tea.KeyTab},
		{"up arrow", tea.KeyUp},
		{"down arrow", tea.KeyDown},
		{"left arrow", tea.KeyLeft},
		{"right arrow", tea.KeyRight},
		{"page up", tea.KeyPgUp},
		{"page down", tea.KeyPgDown},
		{"home", tea.KeyHome},
		{"end", tea.KeyEnd},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := initModel{
				step:      0,
				textInput: "",
				choices:   templates.GetSupportedLanguages(),
			}

			msg := tea.KeyMsg{Type: tt.keyType}
			newModel, _ := model.Update(msg)
			m := newModel.(initModel)

			require.Empty(t, m.textInput, "special key should not be typed as text")
		})
	}
}

// TestUpdate_Backspace_RemovesChar verifies backspace removes the last character
func TestUpdate_Backspace_RemovesChar(t *testing.T) {
	model := initModel{
		step:      0,
		textInput: "test",
		choices:   templates.GetSupportedLanguages(),
	}

	msg := tea.KeyMsg{Type: tea.KeyBackspace}
	newModel, _ := model.Update(msg)
	m := newModel.(initModel)

	require.Equal(t, "tes", m.textInput, "backspace should remove last character")
}

// TestUpdate_Backspace_EmptyInput verifies backspace on empty input doesn't panic
func TestUpdate_Backspace_EmptyInput(t *testing.T) {
	model := initModel{
		step:      0,
		textInput: "",
		choices:   templates.GetSupportedLanguages(),
	}

	msg := tea.KeyMsg{Type: tea.KeyBackspace}
	newModel, _ := model.Update(msg)
	m := newModel.(initModel)

	require.Empty(t, m.textInput, "should remain empty")
}

// TestUpdate_CtrlU_ClearsInput verifies ctrl+u clears the entire input
func TestUpdate_CtrlU_ClearsInput(t *testing.T) {
	model := initModel{
		step:      0,
		textInput: "test input here",
		choices:   templates.GetSupportedLanguages(),
	}

	msg := tea.KeyMsg{Type: tea.KeyCtrlU}
	newModel, _ := model.Update(msg)
	m := newModel.(initModel)

	require.Empty(t, m.textInput, "ctrl+u should clear all input")
}

// TestView_SelectedItemAlignment verifies selected items have consistent alignment
func TestView_SelectedItemAlignment(t *testing.T) {
	model := initModel{
		step:    1,
		cursor:  0,
		choices: templates.GetSupportedLanguages(),
	}

	view := model.View()

	// Check that unselected items start with 2 spaces
	lines := strings.Split(view, "\n")
	for _, line := range lines {
		// Find lines that contain language choices
		for i, choice := range model.choices {
			if strings.Contains(line, choice) && !strings.Contains(line, "Select") {
				if i == model.cursor {
					// Selected item should start with arrow
					require.True(t, strings.HasPrefix(line, "❯ "), "selected item should start with '❯ '")
				} else {
					// Unselected items should start with 2 spaces
					require.True(t, strings.HasPrefix(line, "  "), "unselected item should start with 2 spaces")
				}
			}
		}
	}
}

// TestIsValidProjectName tests project name validation
func TestIsValidProjectName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{"valid lowercase", "myproject", true},
		{"valid with hyphen", "my-project", true},
		{"valid with underscore", "my_project", true},
		{"valid with numbers", "project123", true},
		{"valid complex", "my-awesome-agent_v2", true},
		{"invalid uppercase", "MyProject", false},
		{"invalid starts with number", "123project", false},
		{"invalid with space", "my project", false},
		{"invalid with special chars", "my@project", false},
		{"invalid empty", "", false},
		{"invalid starts with hyphen", "-project", false},
		{"invalid starts with underscore", "_project", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidProjectName(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

// TestIsValidEmail tests email validation
func TestIsValidEmail(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{"valid simple", "user@example.com", true},
		{"valid with dots", "user.name@example.com", true},
		{"valid with plus", "user+tag@example.com", true},
		{"valid with subdomain", "user@mail.example.com", true},
		{"invalid no at", "userexample.com", false},
		{"invalid no domain", "user@", false},
		{"invalid no tld", "user@example", false},
		{"invalid empty", "", false},
		{"invalid just at", "@", false},
		{"invalid double at", "user@@example.com", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidEmail(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

// TestGenerateNodeID tests node ID generation
func TestGenerateNodeID(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple name", "myproject", "myproject"},
		{"with underscore", "my_project", "my-project"},
		{"with multiple underscores", "my__project", "my-project"},
		{"uppercase converted", "MyProject", "myproject"},
		{"trailing hyphen", "project-", "project"},
		{"leading hyphen after conversion", "-project", "project"},
		{"complex", "My_Awesome__Agent-", "my-awesome-agent"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := generateNodeID(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

// TestNormalizeLanguage tests language normalization
func TestNormalizeLanguage(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"python lowercase", "python", "python"},
		{"python short", "py", "python"},
		{"Python capitalized", "Python", "python"},
		{"go lowercase", "go", "go"},
		{"golang", "golang", "go"},
		{"Go capitalized", "Go", "go"},
		{"typescript lowercase", "typescript", "typescript"},
		{"ts short", "ts", "typescript"},
		{"javascript alias", "javascript", "typescript"},
		{"js short", "js", "typescript"},
		{"node alias", "node", "typescript"},
		{"nodejs alias", "nodejs", "typescript"},
		{"unknown", "rust", "rust"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeLanguage(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

// TestGetFileExtension tests file extension lookup
func TestGetFileExtension(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"python", "python", "py"},
		{"go", "go", "go"},
		{"typescript", "typescript", "ts"},
		{"unknown", "rust", "txt"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getFileExtension(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

// TestHandleEnter_StepTransitions tests enter key handling for step transitions
func TestHandleEnter_StepTransitions(t *testing.T) {
	t.Run("step 0 with valid project name advances to step 1", func(t *testing.T) {
		model := initModel{
			step:      0,
			textInput: "myproject",
			choices:   templates.GetSupportedLanguages(),
		}

		newModel, _ := model.handleEnter()
		m := newModel.(initModel)

		require.Equal(t, 1, m.step, "should advance to step 1")
		require.Equal(t, "myproject", m.projectName, "should save project name")
		require.Empty(t, m.textInput, "should clear text input")
	})

	t.Run("step 0 with invalid project name sets error", func(t *testing.T) {
		model := initModel{
			step:      0,
			textInput: "Invalid Project",
			choices:   templates.GetSupportedLanguages(),
		}

		newModel, _ := model.handleEnter()
		m := newModel.(initModel)

		require.Equal(t, 0, m.step, "should stay on step 0")
		require.NotNil(t, m.err, "should set error")
	})

	t.Run("step 1 selects language and advances to step 2", func(t *testing.T) {
		model := initModel{
			step:    1,
			cursor:  1,
			choices: templates.GetSupportedLanguages(),
		}

		newModel, _ := model.handleEnter()
		m := newModel.(initModel)

		require.Equal(t, 2, m.step, "should advance to step 2")
		require.NotEmpty(t, m.language, "should save language")
	})

	t.Run("step 2 with author name advances to step 3", func(t *testing.T) {
		model := initModel{
			step:      2,
			textInput: "John Doe",
			choices:   templates.GetSupportedLanguages(),
		}

		newModel, _ := model.handleEnter()
		m := newModel.(initModel)

		require.Equal(t, 3, m.step, "should advance to step 3")
		require.Equal(t, "John Doe", m.authorName, "should save author name")
	})

	t.Run("step 3 with valid email completes", func(t *testing.T) {
		model := initModel{
			step:      3,
			textInput: "john@example.com",
			choices:   templates.GetSupportedLanguages(),
		}

		newModel, cmd := model.handleEnter()
		m := newModel.(initModel)

		require.True(t, m.done, "should be done")
		require.Equal(t, "john@example.com", m.authorEmail, "should save email")
		require.NotNil(t, cmd, "should return quit command")
	})

	t.Run("step 3 with invalid email sets error", func(t *testing.T) {
		model := initModel{
			step:      3,
			textInput: "invalid-email",
			choices:   templates.GetSupportedLanguages(),
		}

		newModel, _ := model.handleEnter()
		m := newModel.(initModel)

		require.Equal(t, 3, m.step, "should stay on step 3")
		require.NotNil(t, m.err, "should set error")
	})
}

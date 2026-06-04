# Maestro CLI — TUI demo recording (Textual interface).
#
# Generates demo-tui.gif with VHS (https://github.com/charmbracelet/vhs):
#     vhs demo-tui.tape
#
# Needs the [tui] extra:  pip install "maestro-ai-cli[tui]"
# Uses the engine-free examples/demo_plan.yaml (no API keys). Reference it in the
# README's Output Modes / TUI section with:  ![Maestro TUI](demo-tui.gif)

Output demo-tui.gif

Set Shell bash
Set FontSize 16
Set Width 1200
Set Height 740
Set Padding 16
Set Theme "Dracula"
Set WindowBar Colorful
Set TypingSpeed 55ms

Sleep 600ms

# Launch the interactive Textual TUI on the engine-free demo plan
Type "maestro run examples/demo_plan.yaml --output tui"
Enter

# Let the app mount and the tasks run — DAG panel, detail panel, event feed populate
Sleep 5s

# Navigate the DAG panel
Down
Sleep 1500ms
Down
Sleep 1500ms
Enter
Sleep 2s

# Quit the TUI
Type "q"
Sleep 1s

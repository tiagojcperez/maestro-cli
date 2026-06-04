# Maestro CLI — README demo recording (security audit).
#
# Generates demo-audit.gif with VHS (https://github.com/charmbracelet/vhs):
#     vhs docs/assets/demo-audit.tape
#
# Audits examples/risky_plan.yaml (an intentionally dangerous plan) to show that
# a structurally VALID plan can still be UNSAFE — and that `maestro audit` catches
# it. Reference it in README.md with:  ![Maestro audit](docs/assets/demo-audit.gif)

Output docs/assets/demo-audit.gif

Set Shell bash
Set FontSize 18
Set Width 1180
Set Height 680
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set TypingSpeed 55ms

Sleep 600ms

# The plan is structurally valid...
Type "maestro validate examples/risky_plan.yaml"
Enter
Sleep 2500ms

# ...but valid is not the same as safe. Audit it for security risks.
Type "maestro audit examples/risky_plan.yaml"
Enter
Sleep 4s

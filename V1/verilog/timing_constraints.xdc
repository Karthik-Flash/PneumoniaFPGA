# ==============================================================================
# Timing Constraints for Pneumonia Detection Accelerator
# Board  : PYNQ-Z2  (xc7z020clg400-1)
# Target : 80 MHz (12.5 ns period)
#
# PYNQ-Z2 Notes:
#   - Device package is clg400 (not clg484 used on ZedBoard/PYNQ-Z1).
#   - The on-board 125 MHz oscillator drives pin H16 (MRCC capable).
#   - For a pure PL design at 80 MHz, route clk through a MMCM/PLL driven
#     from H16, OR bring 80 MHz in from an external source on a Pmod pin.
#   - For simulation / synthesis timing closure, the create_clock below is
#     sufficient. Assign [get_ports clk] to H16 (or your chosen input pin)
#     in the board-level XDC when moving to physical implementation.
#
# Pin assignments below follow the official PYNQ-Z2 master XDC.
# Uncomment the sections relevant to your physical I/O plan.
# ==============================================================================

# ------------------------------------------------------------------------------
# Primary clock constraint (applies to both synthesis and implementation)
# ------------------------------------------------------------------------------
create_clock -period 12.500 -name clk -waveform {0.000 6.250} [get_ports clk]

# ------------------------------------------------------------------------------
# Input / output delay constraints
# (keeps I/O timing analysis meaningful; adjust for your actual board paths)
# ------------------------------------------------------------------------------
set_input_delay  -clock [get_clocks clk] -min 0.500 [get_ports {rst start pixel_in[*] pixel_valid}]
set_input_delay  -clock [get_clocks clk] -max 2.000 [get_ports {rst start pixel_in[*] pixel_valid}]
set_output_delay -clock [get_clocks clk] -min -1.000 [get_ports {cancer_detected result_valid}]
set_output_delay -clock [get_clocks clk] -max  2.000 [get_ports {cancer_detected result_valid}]

# ------------------------------------------------------------------------------
# False path: async reset (driven from PS or button, not time-critical)
# ------------------------------------------------------------------------------
set_false_path -from [get_ports rst]

# ==============================================================================
# Physical pin assignments — PYNQ-Z2 (xc7z020clg400-1)
# Uncomment and edit when targeting actual hardware.
#
# Option A: Clock from on-board 125 MHz oscillator (needs MMCM to get 80 MHz)
#   set_property PACKAGE_PIN H16          [get_ports clk]
#   set_property IOSTANDARD  LVCMOS33     [get_ports clk]
#
# Option B: Clock from Arduino header pin (IO0 = F14) for external generator
#   set_property PACKAGE_PIN F14          [get_ports clk]
#   set_property IOSTANDARD  LVCMOS33     [get_ports clk]
#
# -- Pmod JA (top row: JA1..JA4 = Y18 Y19 Y16 Y17, bottom: U18 U19 W18 W19)
#   set_property PACKAGE_PIN Y18          [get_ports rst]
#   set_property IOSTANDARD  LVCMOS33     [get_ports rst]
#   set_property PACKAGE_PIN Y19          [get_ports start]
#   set_property IOSTANDARD  LVCMOS33     [get_ports start]
#   set_property PACKAGE_PIN Y16          [get_ports pixel_valid]
#   set_property IOSTANDARD  LVCMOS33     [get_ports pixel_valid]
#   set_property PACKAGE_PIN Y17          [get_ports cancer_detected]
#   set_property IOSTANDARD  LVCMOS33     [get_ports cancer_detected]
#   set_property PACKAGE_PIN U18          [get_ports result_valid]
#   set_property IOSTANDARD  LVCMOS33     [get_ports result_valid]
#
# -- pixel_in[7:0] on Pmod JB (top: W14 Y14 T11 T10, bottom: V16 W16 V12 W13)
#   set_property PACKAGE_PIN W14          [get_ports {pixel_in[0]}]
#   set_property PACKAGE_PIN Y14          [get_ports {pixel_in[1]}]
#   set_property PACKAGE_PIN T11          [get_ports {pixel_in[2]}]
#   set_property PACKAGE_PIN T10          [get_ports {pixel_in[3]}]
#   set_property PACKAGE_PIN V16          [get_ports {pixel_in[4]}]
#   set_property PACKAGE_PIN W16          [get_ports {pixel_in[5]}]
#   set_property PACKAGE_PIN V12          [get_ports {pixel_in[6]}]
#   set_property PACKAGE_PIN W13          [get_ports {pixel_in[7]}]
#   set_property IOSTANDARD  LVCMOS33     [get_ports {pixel_in[*]}]
#
# -- On-board LEDs (LD0..LD3 = R14 P14 N16 M14)
#   set_property PACKAGE_PIN R14          [get_ports cancer_detected]
#   set_property IOSTANDARD  LVCMOS33     [get_ports cancer_detected]
#   set_property PACKAGE_PIN P14          [get_ports result_valid]
#   set_property IOSTANDARD  LVCMOS33     [get_ports result_valid]
#
# -- On-board push-buttons (BTN0..BTN3 = D19 D20 L20 L19)
#   set_property PACKAGE_PIN D19          [get_ports rst]
#   set_property IOSTANDARD  LVCMOS33     [get_ports rst]
#   set_property PACKAGE_PIN D20          [get_ports start]
#   set_property IOSTANDARD  LVCMOS33     [get_ports start]
# ==============================================================================

# %%
"""
Design of Experiment (DoE) for waveguide loss

Authors: Lukas C and Mustafa H)
Date: 2025-02

Description:
- Generates test structures for evaluating spiral delays in waveguides.
- Iteratively adjusts an abstract spiral parameter until the actual spiral length
  is within ±500 µm of the desired target length.
- Each waveguide type gets its own cell containing test structures for various target lengths.
- The cell names now include the actual produced spiral length (in µm).
- Detailed live logging is produced both on console and saved to a log file.
- Waveguide groups (DoE sets) are tiled **vertically** in the top-level cell, 
  and within each group the individual target lengths are placed horizontally.
- Adds a text label at the *center of the spiral instance* on the "Si" layer, with format "widthNm_lengthUm".
  
Dependencies:
- pip install klayout SiEPIC siepic_ebeam_pdk numpy 
"""

import os
import pya
import numpy as np
import logging
from packaging import version

# Set up logging: log to console and file.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler("doe_spiral_tuning.log", mode="w")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# SiEPIC imports
from SiEPIC import __version__ as SiEPIC_version
from SiEPIC.utils.layout import new_layout, floorplan, coupler_array
from SiEPIC.scripts import connect_pins_with_waveguide, connect_cell, export_layout
from SiEPIC.verification import layout_check
from SiEPIC.utils import load_Waveguides_by_Tech

# Validate SiEPIC version
if version.parse(SiEPIC_version) < version.parse("0.5.4"):
    raise Exception("This script requires SiEPIC-Tools version 0.5.4 or greater.")

from SiEPIC._globals import Python_Env

if Python_Env == "Script":
    import siepic_ebeam_pdk
    import dream_ant_lib


class DoE:
    """
    Handles the DoE for evaluating spiral delays in waveguides.
    Each waveguide type gets its own subcell with test structures for various target spiral lengths.
    """

    def __init__(self, designer, tech_name, export_type, waveguides, target_lengths):
        self.designer = designer
        self.tech_name = tech_name
        self.export_type = export_type
        self.waveguides = waveguides
        self.target_lengths = target_lengths  # in µm
        self.cell, self.ly = new_layout(
            tech_name, f"EBeam_{designer}_wgloss", GUI=True, overwrite=True
        )
        self.dbu = self.ly.dbu
        self.waveguide_types = load_Waveguides_by_Tech(tech_name, debug=False)
        logging.info(
            "Initialized DoE with designer '%s' and technology '%s'",
            designer,
            tech_name,
        )

    def validate_waveguide(self, wg):
        available = [wg_data["name"] for wg_data in self.waveguide_types]
        if wg not in available:
            raise ValueError(f"Waveguide type missing: {wg}")
        logging.debug("Waveguide '%s' validated", wg)

    def create_waveguide_cell(self, wg):
        clean_wg = wg.replace(" ", "_").replace(",", "")
        cell_name = f"DoE_{clean_wg}"
        logging.info("Creating waveguide group cell '%s'", cell_name)
        return self.ly.create_cell(cell_name)

    def get_spiral_actual_length(self, cell_delay):
        """
        Extracts the actual spiral length from the delay cell.
        Expects the PCell instance to have a component with a parameter string
        where the first value is formatted as "length=<value>".
        """
        for inst in cell_delay.each_inst():
            try:
                comp_params = inst.cell.find_components()[0].params
                measured = float(comp_params.split(" ")[0].split("=")[1]) * 1e6
                logging.debug("Measured spiral length: %.2f µm", measured)
                return measured
            except Exception as e:
                logging.error("Failed to extract spiral length: %s", str(e))
                raise RuntimeError(
                    "Could not determine spiral length from cell components."
                )
        raise RuntimeError("No instances found in spiral cell.")

    def tune_spiral_for_target(self, wg, target_length, tolerance=500, max_iter=200):
        """
        Iteratively adjusts the abstract 'length' parameter for the spiral PCell
        until the actual spiral length is within ±tolerance (µm) of target_length.
        Uses calibration data:
          - candidate=100  => measured ~8264 µm
          - candidate=200  => measured ~14868 µm
          - candidate=500  => measured ~34068 µm
        We approximate a linear relation:
            measured ≈ 66.04 * candidate + 1660
        A derivative of ~66.04 is used for candidate updates.
        """
        derivative = 66.04
        candidate = target_length / 100  # Initial guess
        logging.info(
            "Tuning spiral for wg='%s' to target %.0f µm; starting candidate: %.2f",
            wg,
            target_length,
            candidate,
        )
        for i in range(max_iter):
            logging.debug("Iteration %d: Trying candidate = %.2f", i + 1, candidate)
            cell_delay = self.ly.create_cell(
                "spiral_paperclip",
                "EBeam_Beta",
                {
                    "waveguide_type": wg,
                    "length": candidate,
                    "loops": 11,
                    "port_vertical": True,
                    "flatten": False,
                },
            )
            try:
                measured = self.get_spiral_actual_length(cell_delay)
            except RuntimeError as err:
                logging.error("Iteration %d: %s", i + 1, str(err))
                raise

            error = target_length - measured
            logging.info(
                "Iteration %d: candidate = %.2f, measured = %.2f µm, error = %.2f µm",
                i + 1,
                candidate,
                measured,
                error,
            )
            if abs(error) <= tolerance:
                logging.info(
                    "Target achieved within tolerance (±%.0f µm) after %d iterations.",
                    tolerance,
                    i + 1,
                )
                return cell_delay, measured
            if measured == 0:
                logging.error(
                    "Iteration %d: Measured length is zero. Aborting iteration.", i + 1
                )
                break
            # Update candidate using a derivative-based correction:
            new_candidate = candidate + (target_length - measured) / derivative
            logging.debug(
                "Iteration %d: Updating candidate from %.2f to %.2f",
                i + 1,
                candidate,
                new_candidate,
            )
            candidate = new_candidate
        raise RuntimeError(
            f"Could not tune spiral to target length {target_length} µm within {max_iter} iterations."
        )

    def create_test_structure(self, parent_cell, wg, target_length):
        """
        Creates a test structure for a given waveguide type and target spiral length.
        The test structure cell is named using the actual produced spiral length.
        Includes couplers, a taper between the GC and the spiral, and a tuned spiral delay.

        Each test structure is placed horizontally within the waveguide group cell.
        Also places a text label at the *center of the spiral instance* on the "Si" layer,
        with format "<widthInNm>nm_<spiralLength>um".
        """
        clean_wg = wg.replace(" ", "_").replace(",", "")
        try:
            cell_delay, measured = self.tune_spiral_for_target(wg, target_length)
        except RuntimeError as e:
            logging.error("Tuning spiral failed: %s", str(e))
            raise

        # Use actual measured length in cell name and device label
        subcell_name = f"spiral_{clean_wg}_{int(measured)}um"
        logging.info("Creating test structure '%s'", subcell_name)
        subcell = self.ly.create_cell(subcell_name)

        # Determine wavelength from wg string and set GC port width accordingly
        wavelength = 1550 if "1550" in wg else 1310
        gc_width = 0.500 if wavelength == 1550 else 0.350

        # Parse spiral port width from the waveguide string (e.g., "w=300 nm")
        try:
            spiral_width_nm = int(wg.split("w=")[1].split(" ")[0])  # e.g. 300
            spiral_width_db = spiral_width_nm * self.dbu
        except Exception as e:
            logging.error("Failed to parse spiral width from '%s': %s", wg, str(e))
            raise

        # Coupler label includes measured length
        coupler_label = f"opt_in_TE_{wavelength}_device_{self.designer}_{clean_wg}_{int(measured)}um"
        inst_couplers = coupler_array(
            subcell,
            x_offset=0,
            y_offset=(127e3) / 2,
            count=2,
            label=coupler_label,
            label_location=2,
            cell_name=f"GC_TE_{wavelength}_8degOxide_BB",
            cell_library="EBeam",
            cell_params={"wavelength": wavelength},
        )
        logging.info("Coupler array created for wavelength %d nm", wavelength)

        # Taper to match GC port width to spiral width
        taper_params = {
            "wg_length": 5,
            "wg_width1": gc_width,
            "wg_width2": spiral_width_db,
        }
        taper_cell = self.ly.create_cell("ebeam_taper_te1550", "EBeam", taper_params)
        logging.info("Created taper with parameters: %s", taper_params)

        # Connect top GC to taper, then taper to spiral
        inst_taper1 = connect_cell(inst_couplers[0], "opt1", taper_cell, "pin1")
        inst_delay = connect_cell(inst_taper1, "pin2", cell_delay, "opt1")
        logging.info("Connected top GC to taper and taper to spiral delay.")

        # Connect spiral to second coupler via the same taper
        inst_taper2 = connect_cell(inst_couplers[1], "opt1", taper_cell, "pin1")
        connect_pins_with_waveguide(
            inst_delay,
            "opt2",
            inst_taper2,
            "pin2",
            waveguide_type=wg,
            turtle_A=[15, -90, 15, 90],
            turtle_B=[15, 90, 30, 90, 0, -90],
        )
        logging.info("Connected spiral delay output to second coupler via taper.")

        # Place subcell horizontally
        trans = pya.Trans(pya.Trans.R0, parent_cell.bbox().right + 45e3, 0)
        parent_cell.insert(pya.CellInstArray(subcell.cell_index(), trans))
        logging.debug("Test structure '%s' inserted into parent cell", subcell_name)

        # ---- Add text label in its own cell, enclosed by a DevRec box ----
        label_wavl = "C" if wavelength == 1550 else "O"
        label_text = f"{spiral_width_nm}nm_{int(measured)}um_{label_wavl}"
        # Create text cell using the given method (do not use TECHNOLOGY.get)
        layer_si = self.ly.TECHNOLOGY["Si"]
        text_params = {
            "text": label_text,
            "layer": layer_si,
            "mag": 10,
            "inverse": False,
        }
        text_cell = self.ly.create_cell("TEXT", "Basic", text_params)
        # Create a new empty cell to serve as a container for the text label
        label_container = self.ly.create_cell("LABEL_" + label_text)
        # Insert the text cell instance into the label container (without transformation)
        label_container.insert(pya.CellInstArray(text_cell.cell_index(), pya.Trans()))
        # Enclose the text with a DevRec box
        # Get bounding box of the text cell (the instance in the container may be at (0,0))
        bbox_text = text_cell.bbox()
        margin = 10
        box_label = pya.Box(bbox_text.left - margin, bbox_text.bottom - margin,
                             bbox_text.right + margin, bbox_text.top + margin)
        # Directly use self.ly.TECHNOLOGY["DevRec"] (do not use get)
        devrec_layer = self.ly.TECHNOLOGY["DevRec"]
        label_container.shapes(devrec_layer).insert(box_label)
        # Determine the center of the spiral instance (using inst_delay's bounding box)
        spiral_box = inst_delay.bbox()
        extra_y = 52000
        center_x = (spiral_box.left + spiral_box.right) // 2 + 1000
        center_y = (spiral_box.bottom + spiral_box.top) // 2 + extra_y
        # Use the dimensions of the label container
        label_width = label_container.bbox().width()
        label_height = label_container.bbox().height()
        label_trans = pya.Trans(
            pya.Trans.R0,
            center_x - label_width // 2,
            center_y - label_height // 2,
        )
        # Insert the label container instance into the subcell
        subcell.insert(pya.CellInstArray(label_container.cell_index(), label_trans))
        logging.info(
            "Added text label '%s' (in cell '%s') at center of spiral instance in subcell '%s'.",
            label_text,
            label_container.name,
            subcell_name,
        )

    def generate_test_structures(self):
        """
        Generates test structures for each waveguide type and each target spiral length.
        Tiling each waveguide group cell **vertically** in the top-level cell.
        """
        y_offset = 0  # track vertical offset in the top-level cell
        for wg in self.waveguides:
            logging.info("Processing waveguide: '%s'", wg)
            self.validate_waveguide(wg)
            wg_cell = self.create_waveguide_cell(wg)

            # Create the sub-structures horizontally within wg_cell
            for target_length in self.target_lengths:
                logging.info(
                    "Generating test structure for target length: %.0f µm",
                    target_length,
                )
                self.create_test_structure(wg_cell, wg, target_length)

            # Insert waveguide group cell into top-level cell at y_offset
            trans = pya.Trans(pya.Trans.R0, 0, y_offset)
            self.cell.insert(pya.CellInstArray(wg_cell.cell_index(), trans))
            logging.debug(
                "Waveguide group cell '%s' inserted into top-level cell at y_offset=%.1f",
                wg,
                y_offset,
            )

            # Move up for the next waveguide group
            box = wg_cell.bbox()
            group_height = box.height()
            margin = 20e3  # some vertical spacing
            y_offset += group_height + margin

    def define_floorplan(self):
        w = self.cell.bbox().right + 10e3
        h = self.cell.bbox().top + 10e3
        logging.info("Defining floorplan size: %.0f x %.0f", w, h)
        floorplan(self.cell, w, h)

    def export_layout(self):
        path = os.path.dirname(os.path.realpath(__file__))
        filename, _ = os.path.splitext(os.path.basename(__file__))
        if self.export_type == "static":
            file_out = export_layout(self.cell, path, filename, format="oas", screenshot=True)
        else:
            file_out = os.path.join(path, f"{filename}.oas")
            self.ly.write(file_out)
        logging.info("Exported layout to '%s'", file_out)
        return file_out

    def run_verification(self):
        path = os.path.dirname(os.path.realpath(__file__))
        filename, _ = os.path.splitext(os.path.basename(__file__))
        file_lyrdb = os.path.join(path, f"{filename}.lyrdb")
        num_errors = layout_check(cell=self.cell, verbose=False, GUI=True, file_rdb=file_lyrdb)
        logging.info("Layout verification completed with %d errors", num_errors)

    def display_layout(self, file_out):
        from SiEPIC.utils import klive
        logging.info("Displaying layout from '%s'", file_out)
        klive.show(file_out, technology=self.tech_name)


def main():
    waveguides_DoE = [
        "Strip TE 1550 nm, w=300 nm",
        "Strip TE 1310 nm, w=300 nm",
        "Strip TE 1550 nm, w=350 nm",
        "Strip TE 1310 nm, w=350 nm",
        "Strip TE 1550 nm, w=400 nm",
        "Strip TE 1310 nm, w=400 nm",
        "Strip TE 1550 nm, w=450 nm",
        "Strip TE 1310 nm, w=450 nm",
        "Strip TE 1550 nm, w=500 nm",
        "Strip TE 1310 nm, w=500 nm",
        "Strip TE 1550 nm, w=550 nm",
        "Strip TE 1310 nm, w=550 nm",
        "Strip TE 1550 nm, w=600 nm",
        "Strip TE 1310 nm, w=600 nm",
    ]
    target_lengths = [6000, 12000, 24000, 48000]

    logging.info("Starting DoE generation")
    doe = DoE(
        designer="Mustafa",
        tech_name="EBeam",
        export_type="static",
        waveguides=waveguides_DoE,
        target_lengths=target_lengths,
    )
    doe.generate_test_structures()
    doe.define_floorplan()
    file_out = doe.export_layout()
    doe.run_verification()
    doe.display_layout(file_out)
    logging.info("DoE generation completed.")


if __name__ == "__main__":
    main()

# %%

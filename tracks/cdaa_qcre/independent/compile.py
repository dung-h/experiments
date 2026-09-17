"""
Compile and optimize the original test suite circuits using SQGM and SABRE in the old
environment and TKET and Qiskit in the new environment. Automatically detects which
environment it has been run in using the installed Python packages.
"""

import sys
import subprocess
import os
import shutil

from importlib_metadata import version


if __name__ == "__main__":

    # Determine whether we are in the "old" or "new" compilation environment
    # Old is used to compile with SABRE and SQGM, new is used for TKET and Qiskit
    try:
        ver_num = version("qiskit")
    except:
        print("No qiskit package found, looking for package 'qiskit-terra'.")

        try:
            ver_num = version("qiskit-terra")
        except:
            print("No qiskit-terra package found, aborting.")
            sys.exit(1)
        else:
            comp_env = "old"
            print(
                f"Found qiskit-terra {ver_num}, entering 'old' compilation environment."
            )
    else:
        comp_env = "new"
        print(f"Found qiskit {ver_num}, entering 'new' compilation environment.")

    if comp_env == "old":

        # Configuration
        hardware = {
            "eagle": [
                "ibm_sherbrooke",
                "ibm_kyiv",
                "ibm_brisbane",
            ],
            "heron": [
                "ibm_marrakesh",
                "ibm_kingston",
                "ibm_aachen",
            ],
        }
        compilers = [
            "sqgm",
            "sabre0330",
        ]

        for architecture, devices in hardware.items():
            print(f"+++{architecture}+++")
            print(
                "Starting sqgm artifact script with configuration "
                + f"'sqgm/exp-in/exp_{architecture}.json'."
            )

            # Ue pre-existing sqgm script to compile with (older) sqgm and sabre0330 algorithms
            # Ignore warnings because Qiskit Terra raises many deprecation errors
            proc = subprocess.Popen(
                (
                    "python",
                    "-W",
                    "ignore",
                    "sqgm/main.py",
                    f"sqgm/exp-in/exp_{architecture}.json",
                )
            )
            proc.wait()

            comp_dir = r"./qasm/compiled/"

            # Copy compiled .qasm files to matching compiler, architecture, and device directory
            #
            # sqgm script exports .qasm files to one target directory. Sorting from this target
            # directory to the matching /compiler/architecture/device directory allows us to
            # maintain consistent file organization to simplify the next phase in the methodology.
            for file_name in os.listdir(comp_dir):
                if file_name.endswith(".qasm"):
                    full_file_path = os.path.join(comp_dir, file_name)

                    # Delete uncompiled and input files (only keep compiled ones)
                    if "init" in file_name or "min-in" in file_name:
                        os.remove(full_file_path)
                        continue

                    # Copy and rename compiled .qasms to correct directory
                    for compiler in compilers:
                        if compiler in file_name:
                            file_name_pieces = file_name.split("-")
                            new_file_name = file_name_pieces[0] + ".qasm"

                            for device in devices:
                                new_file_path = rf"./qasm/compiled/{compiler}/{architecture}/{device}/{new_file_name}"
                                shutil.copy(full_file_path, new_file_path)

                    # Delete copied files after reading
                    os.remove(full_file_path)

        sys.exit(0)

    elif comp_env == "new":

        from util import comp_qiskit
        from util import comp_tket

        # Configuration
        compiler_funcs = {
            "qiskit141": comp_qiskit,
            "tket": comp_tket,
        }
        hardware = {
            "eagle": [
                "ibm_sherbrooke",
                "ibm_kyiv",
                "ibm_brisbane",
            ],
            "heron": [
                "ibm_marrakesh",
                "ibm_kingston",
                "ibm_aachen",
            ],
        }
        num_reps = 5

        # Compile using qiskit141 and tket for each architecture and device
        for compiler, compile_func in compiler_funcs.items():

            print(f"==={compiler}===")
            for architecture, device_names in hardware.items():

                print(f"+++{architecture}+++")

                for device_name in device_names:

                    print(f"---{device_name}---")

                    in_qasm_dir = r"./qasm/original/"
                    out_qasm_dir = (
                        rf"./qasm/compiled/{compiler}/{architecture}/{device_name}/"
                    )

                    compile_func(
                        architecture, device_name, num_reps, in_qasm_dir, out_qasm_dir
                    )

        sys.exit(0)

    # If comp_env is neither "old" or "new", abort
    else:
        print("Compilation environment could not be established, aborting.")
        sys.exit(1)

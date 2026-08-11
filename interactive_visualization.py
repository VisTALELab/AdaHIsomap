import os
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def select_Adamapper_parameters_interactively(
    data_type,
    dataset_type,
    round_number,
    n_samples,
    current_parameters,
    default_parameters,
    available_filter_functions,
    available_base_point_methods,
):
    """
    Open an interactive window for selecting parameters shared by
    Mapper and AdaMapper.

    Round behavior
    --------------
    Round 1:
        current_parameters contains the recommended default values.

    Round 2 and later:
        current_parameters contains the parameters used in the
        previous round.

    The recommended defaults are always displayed in a separate
    column. The user can restore them using the
    "Use Recommended Defaults" button.

    Base-point behavior
    -------------------
    A base-point method is required when:

        filter_function == "base_point_geodesic_distance"

    For filter functions that do not use a base point, BP may be None.

    Parameters
    ----------
    data_type : str
        Name of the current dataset.

    round_number : int
        Current experiment round, beginning at 1.

    n_samples : int
        Number of data points in the current dataset.

        This is used to validate:

            1 <= n_neighbors < n_samples

    current_parameters : dict
        Values initially displayed in the editable fields.

        Required structure:

            {
                "filter_function": str,
                "overlap_perc": float,
                "BP": str or None,
                "n_neighbors": int,
                "min_samples": int,
            }

    default_parameters : dict
        Recommended values for the current dataset.

        This dictionary must have the same structure as
        current_parameters.

    available_filter_functions : sequence of str
        Filter functions that the user may select.

    available_base_point_methods : sequence
        Base-point methods the user may select.

        Example:

            ["EP", "DR", "BC", None]

    Returns
    -------
    dict
        When the user runs the experiment:

            {
                "action": "run",
                "parameters": {
                    "filter_function": str,
                    "overlap_perc": float,
                    "BP": str or None,
                    "n_neighbors": int,
                    "min_samples": int,
                },
            }

        When the user quits or closes the window:

            {
                "action": "quit",
                "parameters": None,
            }
    """

    if dataset_type not in {"standard", "network"}:
        raise ValueError(
            "dataset_type must be either "
            "'standard' or 'network'.")

    # ==================================================
    # 1. VALIDATE INPUTS FROM THE MAIN PROGRAM
    # ==================================================

    required_parameter_names = {
        "filter_function",
        "overlap_perc",
        "BP",
        "n_neighbors",
        "min_samples",
    }

    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError(
            "n_samples must be an integer greater than or equal to 2, "
            f"but received {n_samples!r}."
        )

    missing_current_parameters = (
        required_parameter_names - set(current_parameters)
    )

    if missing_current_parameters:
        raise ValueError(
            "current_parameters is missing the following value(s): "
            f"{sorted(missing_current_parameters)}"
        )

    missing_default_parameters = (
        required_parameter_names - set(default_parameters)
    )

    if missing_default_parameters:
        raise ValueError(
            "default_parameters is missing the following value(s): "
            f"{sorted(missing_default_parameters)}"
        )

    available_filter_functions = list(
        available_filter_functions
    )

    available_base_point_methods = list(
        available_base_point_methods
    )

    if dataset_type == "network":

        available_filter_functions = ["base_point_geodesic_distance"]
        available_base_point_methods = ["EP", "BC"]


    if not available_filter_functions:
        raise ValueError(
            "available_filter_functions cannot be empty."
        )

    if not available_base_point_methods:
        raise ValueError(
            "available_base_point_methods cannot be empty."
        )

    if (
        current_parameters["filter_function"]
        not in available_filter_functions
    ):
        raise ValueError(
            "The current filter function "
            f"{current_parameters['filter_function']!r} is not present "
            "in available_filter_functions."
        )

    if (
        default_parameters["filter_function"]
        not in available_filter_functions
    ):
        raise ValueError(
            "The default filter function "
            f"{default_parameters['filter_function']!r} is not present "
            "in available_filter_functions."
        )

    if (
        current_parameters["BP"]
        not in available_base_point_methods
    ):
        raise ValueError(
            "The current base-point method "
            f"{current_parameters['BP']!r} is not present in "
            "available_base_point_methods."
        )

    if (
        default_parameters["BP"]
        not in available_base_point_methods
    ):
        raise ValueError(
            "The default base-point method "
            f"{default_parameters['BP']!r} is not present in "
            "available_base_point_methods."
        )

    # ==================================================
    # 2. DEFAULT DECISION
    # ==================================================
    #
    # If the user closes the window using X, this value
    # remains "quit".
    # ==================================================

    decision = {
        "action": "quit",
        "parameters": None,
    }

    # ==================================================
    # 3. CREATE THE WINDOW
    # ==================================================

    root = tk.Tk()

    root.title(
        f"Parameter Selection — {data_type} — Round {round_number}"
    )

    root.resizable(False, False)

    # ==================================================
    # 4. VALUES CONNECTED TO THE GUI CONTROLS
    # ==================================================

    filter_function_var = tk.StringVar(
        value=str(current_parameters["filter_function"])
    )

    overlap_perc_var = tk.StringVar(
        value=str(current_parameters["overlap_perc"])
    )

    # A Tkinter Combobox displays text. Therefore, Python None
    # is displayed as the text "None".
    current_bp_display = (
        "None"
        if current_parameters["BP"] is None
        else str(current_parameters["BP"])
    )

    base_point_var = tk.StringVar(
        value=current_bp_display
    )

    if dataset_type == "standard":
        n_neighbors_var = tk.StringVar(
            value=str(current_parameters["n_neighbors"])
        )

    else:
        n_neighbors_var = None




    min_samples_var = tk.StringVar(
        value=str(current_parameters["min_samples"])
    )

    # Convert each available BP option into text for the Combobox.
    base_point_display_values = [
        "None" if method is None else str(method)
        for method in available_base_point_methods
    ]

    # ==================================================
    # 5. MAIN CONTAINER
    # ==================================================

    main_frame = ttk.Frame(
        root,
        padding=(24, 20),
    )

    main_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
    )

    # ==================================================
    # 6. TITLE AND INSTRUCTIONS
    # ==================================================

    title_label = ttk.Label(
        main_frame,
        text=f"Select Parameters for {data_type}",
        font=("TkDefaultFont", 15, "bold"),
    )

    title_label.grid(
        row=0,
        column=0,
        columnspan=3,
        pady=(0, 5),
    )

    round_label = ttk.Label(
        main_frame,
        text=f"Experiment Round {round_number}",
        font=("TkDefaultFont", 11, "bold"),
    )

    round_label.grid(
        row=1,
        column=0,
        columnspan=3,
        pady=(0, 12),
    )

    if round_number == 1:
        instruction_message = (
            "The fields below contain the recommended default "
            "parameters. You may change any value before running."
        )
    else:
        instruction_message = (
            "The editable fields contain the parameters used in the "
            "previous round. Recommended defaults are shown on the right."
        )

    instruction_label = ttk.Label(
        main_frame,
        text=instruction_message,
        wraplength=650,
        justify="center",
    )

    instruction_label.grid(
        row=2,
        column=0,
        columnspan=3,
        pady=(0, 18),
    )

    # ==================================================
    # 7. TABLE HEADINGS
    # ==================================================

    ttk.Label(
        main_frame,
        text="Parameter",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=3,
        column=0,
        sticky="w",
        padx=(0, 18),
        pady=(0, 8),
    )

    ttk.Label(
        main_frame,
        text="Parameters for This Round",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=3,
        column=1,
        sticky="w",
        padx=(0, 22),
        pady=(0, 8),
    )

    ttk.Label(
        main_frame,
        text="Recommended Default",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=3,
        column=2,
        sticky="w",
        pady=(0, 8),
    )

    # ==================================================
    # 8. HELPER FOR DEFAULT-VALUE LABELS
    # ==================================================

    def create_default_label(row, value):
        display_value = (
            "None"
            if value is None
            else str(value)
        )

        label = ttk.Label(
            main_frame,
            text=display_value,
        )

        label.grid(
            row=row,
            column=2,
            sticky="w",
            pady=6,
        )

        return label

    # ==================================================
    # 9. FILTER FUNCTION
    # ==================================================

    ttk.Label(
        main_frame,
        text="Filter function",
    ).grid(
        row=4,
        column=0,
        sticky="w",
        padx=(0, 18),
        pady=6,
    )

    filter_function_combobox = ttk.Combobox(
        main_frame,
        textvariable=filter_function_var,
        values=available_filter_functions,
        state="readonly",
        width=34,
    )

    filter_function_combobox.grid(
        row=4,
        column=1,
        sticky="ew",
        padx=(0, 22),
        pady=6,
    )

    create_default_label(
        row=4,
        value=default_parameters["filter_function"],
    )

    # ==================================================
    # 10. OVERLAP PERCENTAGE
    # ==================================================

    ttk.Label(
        main_frame,
        text="Overlap percentage",
    ).grid(
        row=5,
        column=0,
        sticky="w",
        padx=(0, 18),
        pady=6,
    )

    overlap_entry = ttk.Entry(
        main_frame,
        textvariable=overlap_perc_var,
        width=37,
    )

    overlap_entry.grid(
        row=5,
        column=1,
        sticky="ew",
        padx=(0, 22),
        pady=6,
    )

    create_default_label(
        row=5,
        value=default_parameters["overlap_perc"],
    )

    # ==================================================
    # 11. BASE-POINT METHOD
    # ==================================================

    ttk.Label(
        main_frame,
        text="Base-point method",
    ).grid(
        row=6,
        column=0,
        sticky="w",
        padx=(0, 18),
        pady=6,
    )

    base_point_combobox = ttk.Combobox(
        main_frame,
        textvariable=base_point_var,
        values=base_point_display_values,
        state="readonly",
        width=34,
    )

    base_point_combobox.grid(
        row=6,
        column=1,
        sticky="ew",
        padx=(0, 22),
        pady=6,
    )

    create_default_label(
        row=6,
        value=default_parameters["BP"],
    )

    # ==================================================
    # 12. NUMBER OF NEIGHBORS
    # ==================================================

    if dataset_type == "standard":

        ttk.Label(
            main_frame,
            text="Number of neighbors",
        ).grid(
            row=7,
            column=0,
            sticky="w",
            padx=(0, 18),
            pady=6,
        )

        n_neighbors_entry = ttk.Entry(
            main_frame,
            textvariable=n_neighbors_var,
            width=37,
        )

        n_neighbors_entry.grid(
            row=7,
            column=1,
            sticky="ew",
            padx=(0, 22),
            pady=6,
        )

        create_default_label(
            row=7,
            value=default_parameters["n_neighbors"],
        )

    # ==================================================
    # 13. MINIMUM SAMPLES
    # ==================================================

    ttk.Label(
        main_frame,
        text="Minimum samples",
    ).grid(
        row=8,
        column=0,
        sticky="w",
        padx=(0, 18),
        pady=6,
    )

    min_samples_entry = ttk.Entry(
        main_frame,
        textvariable=min_samples_var,
        width=37,
    )

    min_samples_entry.grid(
        row=8,
        column=1,
        sticky="ew",
        padx=(0, 22),
        pady=6,
    )

    create_default_label(
        row=8,
        value=default_parameters["min_samples"],
    )

    # ==================================================
    # 14. EXPLANATION NOTE
    # ==================================================

    if dataset_type == "network":

        note_message = (
            "\nNetwork datasets use only "
            "base_point_geodesic_distance.\n"
            "Available base-point methods are EP and BC.\n\n"
            "Use Recommended Defaults only resets the fields. "
            "Click Run with These Parameters to start the round."
        )

    else:

        note_message = (
            "\nOnly base_point_geodesic_distance requires EP, DR, or BC. "
            "For all other filter functions, select None.\n\n"
            "Use Recommended Defaults only resets the fields. "
            "Click Run with These Parameters to start the round."
        )


    note_label = ttk.Label(
        main_frame,
        text=note_message,
        wraplength=650,
        justify="left",
    )


    note_label.grid(
        row=9,
        column=0,
        columnspan=3,
        sticky="w",
        pady=(14, 14),
    )

    # ==================================================
    # 15. BUTTON CONTAINER
    # ==================================================

    button_frame = ttk.Frame(
        main_frame
    )

    button_frame.grid(
        row=10,
        column=0,
        columnspan=3,
        pady=(4, 0),
    )

    # ==================================================
    # 16. RESTORE DEFAULTS
    # ==================================================

    def restore_defaults():
        """
        Restore the recommended values without closing the window.
        """

        filter_function_var.set(
            str(default_parameters["filter_function"])
        )

        overlap_perc_var.set(
            str(default_parameters["overlap_perc"])
        )

        default_bp = default_parameters["BP"]

        base_point_var.set(
            "None"
            if default_bp is None
            else str(default_bp)
        )


        if dataset_type == "standard":

            n_neighbors_var.set(
                str(default_parameters["n_neighbors"]))


        min_samples_var.set(
            str(default_parameters["min_samples"])
        )

    # ==================================================
    # 17. VALIDATE AND ACCEPT PARAMETERS
    # ==================================================

    def run_with_parameters():
        """
        Validate the displayed values and close the window when valid.
        """

        filter_function = (
            filter_function_var.get().strip()
        )

        base_point_display = (
            base_point_var.get().strip()
        )

        # Convert the displayed text back into Python None.
        if base_point_display == "None":
            base_point_method = None
        else:
            base_point_method = base_point_display

        # ==================================================

        if dataset_type == "network":

            if filter_function != "base_point_geodesic_distance":

                messagebox.showerror(
                    "Invalid Network Filter",
                    "Network datasets require "
                    "base_point_geodesic_distance.",
                    parent=root)
                return

            if base_point_method not in {"EP", "BC"}:

                messagebox.showerror(
                    "Invalid Network Base Point",
                    "Network datasets support only EP or BC.",
                    parent=root)
                
                return
            
        # ----------------------------------------------
        # Filter-function validation
        # ----------------------------------------------

        if filter_function not in available_filter_functions:
            messagebox.showerror(
                "Invalid Filter Function",
                "Please select a filter function from the available list.",
                parent=root,
            )
            return

        # ----------------------------------------------
        # Base-point validation
        # ----------------------------------------------

        if base_point_method not in available_base_point_methods:
            messagebox.showerror(
                "Invalid Base-Point Method",
                "Please select a base-point method from the available list.",
                parent=root,
            )
            return


        if (
            filter_function == "base_point_geodesic_distance"
            and base_point_method is None
        ):

            if dataset_type == "network":
                base_point_message = (
                    "The base_point_geodesic_distance filter requires a "
                    "base-point method. Please select EP or BC."
                )
            else:
                base_point_message = (
                    "The base_point_geodesic_distance filter requires a "
                    "base-point method. Please select EP, DR, or BC."
                )

            messagebox.showerror(
                "Base Point Required",
                base_point_message,
                parent=root,
            )

            base_point_combobox.focus_set()
            return
        
        # ----------------------------------------------
        # Overlap-percentage validation
        # ----------------------------------------------

        try:
            overlap_perc = float(
                overlap_perc_var.get().strip()
            )
        except ValueError:
            messagebox.showerror(
                "Invalid Overlap Percentage",
                "Overlap percentage must be a numerical value.",
                parent=root,
            )
            overlap_entry.focus_set()
            return

        if not 0.0 <= overlap_perc < 1.0:
            messagebox.showerror(
                "Invalid Overlap Percentage",
                "Overlap percentage must satisfy "
                "0 ≤ overlap_perc < 1.",
                parent=root,
            )
            overlap_entry.focus_set()
            return

        # ----------------------------------------------
        # Number-of-neighbors validation
        # ----------------------------------------------
       
        if dataset_type == "standard":

            try:
                n_neighbors = int(
                    n_neighbors_var.get().strip()
                )

            except ValueError:
                messagebox.showerror(
                    "Invalid Number of Neighbors",
                    "Number of neighbors must be an integer.",
                    parent=root,
                )
                n_neighbors_entry.focus_set()
                return

            if n_neighbors < 1:
                messagebox.showerror(
                    "Invalid Number of Neighbors",
                    "Number of neighbors must be at least 1.",
                    parent=root,
                )
                n_neighbors_entry.focus_set()
                return

            if n_neighbors >= n_samples:
                messagebox.showerror(
                    "Invalid Number of Neighbors",
                    "Number of neighbors must be smaller than the number "
                    f"of data points.\n\n"
                    f"Number of data points: {n_samples}\n"
                    f"Entered n_neighbors: {n_neighbors}",
                    parent=root,
                )
                n_neighbors_entry.focus_set()
                return
        
        else:

            n_neighbors = None

        # ----------------------------------------------
        # Minimum-samples validation
        # ----------------------------------------------

        try:
            min_samples = int(
                min_samples_var.get().strip()
            )
        except ValueError:
            messagebox.showerror(
                "Invalid Minimum Samples",
                "Minimum samples must be an integer.",
                parent=root,
            )
            min_samples_entry.focus_set()
            return

        if min_samples < 1:
            messagebox.showerror(
                "Invalid Minimum Samples",
                "Minimum samples must be at least 1.",
                parent=root,
            )
            min_samples_entry.focus_set()
            return

        # ----------------------------------------------
        # Save the accepted values
        # ----------------------------------------------

        decision["action"] = "run"

        decision["parameters"] = {
            "filter_function": filter_function,
            "overlap_perc": overlap_perc,
            "BP": base_point_method,
            "n_neighbors": n_neighbors,
            "min_samples": min_samples,
        }

        root.destroy()

    # ==================================================
    # 18. QUIT CALLBACK
    # ==================================================

    def quit_program():
        decision["action"] = "quit"
        decision["parameters"] = None
        root.destroy()

    # ==================================================
    # 19. BUTTONS
    # ==================================================

    run_button = ttk.Button(
        button_frame,
        text="Run with These Parameters",
        command=run_with_parameters,
    )

    run_button.grid(
        row=0,
        column=0,
        padx=6,
    )

    defaults_button = ttk.Button(
        button_frame,
        text="Use Recommended Defaults",
        command=restore_defaults,
    )

    defaults_button.grid(
        row=0,
        column=1,
        padx=6,
    )

    quit_button = ttk.Button(
        button_frame,
        text="Quit Program",
        command=quit_program,
    )

    quit_button.grid(
        row=0,
        column=2,
        padx=6,
    )

    # ==================================================
    # 20. WINDOW AND KEYBOARD BEHAVIOR
    # ==================================================

    root.protocol(
        "WM_DELETE_WINDOW",
        quit_program,
    )

    main_frame.columnconfigure(
        1,
        weight=1,
    )

    # ==================================================
    # 21. CENTER THE WINDOW
    # ==================================================

    root.update_idletasks()

    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_x = max(
        0,
        (screen_width - window_width) // 2,
    )

    window_y = max(
        0,
        (screen_height - window_height) // 2,
    )

    root.geometry(
        f"{window_width}x{window_height}+{window_x}+{window_y}"
    )

    root.lift()
    root.focus_force()

    # Wait here until the user closes the window.
    root.mainloop()

    return decision




def select_persistence_threshold_interactively(
    features,
    default_T=0.35,
    round_number=1,
    data_type="dataset",
    dataset_type="standard",
    output_dir=None,
    output_prefix=None):

    """
    Display an interactive persistence diagram and allow the user
    to select the persistence-threshold ratio T.

    The absolute persistence cutoff is:

        cutoff = T * max_persistence

    An H1 feature is retained when:

        persistence_value >= cutoff

    Threshold interpretation
    ------------------------
    1. All H1 features retained:
       No persistence-based simplification is applied.
       AdaMapper is used.

    2. Some H1 features retained:
       Persistence-based simplification is applied.
       AdaMapper is used.

    3. No H1 features retained:
       Standard Mapper is used automatically.

    Parameters
    ----------
    features : list of dict
        All H1 feature dictionaries returned from Julia.

        The structure of this list and its dictionaries is not changed.

    default_T : float, default=0.35
        Paper-defined default persistence-threshold ratio.

        For example, T=0.35 produces a cutoff equal to 35% of
        the maximum persistence.

    round_number : int, default=1
        Current experiment round. Used in the saved output filename.

    data_type : str, default="dataset"
        Dataset name. Used in the window title and output filename.

    output_prefix : str, optional

    Filename prefix for this experiment round.
        Example:
            "1-Fertility"

    Returns
    -------
    selected_features : list of dict
        H1 features retained by the selected threshold.

        This remains a list containing dictionaries from the original
        features list.

    selected_T : float or None
        Selected persistence-threshold ratio.

        This is None only when no H1 features exist.

    selected_cutoff : float or None
        Selected absolute persistence cutoff.

        This is None only when no H1 features exist.

    algorithm_mode : str
        Either:

            "adamapper"
            "standard_mapper"

    threshold_action : str
        Either:

            "run"
            "quit"
    """

    # --------------------------------------------------
    # Validate general arguments
    # --------------------------------------------------
    if not isinstance(round_number, int) or round_number < 1:
        raise ValueError(
            "round_number must be an integer greater than or equal to 1."
        )

    if not isinstance(data_type, str) or not data_type.strip():
        raise ValueError(
            "data_type must be a nonempty string."
        )

    if dataset_type not in {"standard", "network"}:
        raise ValueError(
            "dataset_type must be either "
            "'standard' or 'network'.")
    # --------------------------------------------------
    # No H1 features exist
    # --------------------------------------------------

    if len(features) == 0:

        if dataset_type == "network":
            raise ValueError(
                "The network dataset contains no H1 features. "
                "AdaMapper requires at least one retained H1 feature.")

        print("\nNo H1 features were found.")
        print("Standard Mapper should be used.")

        return [], None, None, "standard_mapper", "run"

    # --------------------------------------------------
    # Extract persistence information
    # --------------------------------------------------
    births = np.array(
        [
            feature["birth_value"]
            for feature in features
        ],
        dtype=float,
    )

    deaths = np.array(
        [
            feature["Death_Time"]
            for feature in features
        ],
        dtype=float,
    )

    persistences = np.array(
        [
            feature["persistence_value"]
            for feature in features
        ],
        dtype=float,
    )

    # --------------------------------------------------
    # Validate persistence data
    # --------------------------------------------------
    if not np.all(np.isfinite(births)):
        raise ValueError(
            "The birth values contain NaN or infinite values."
        )

    if not np.all(np.isfinite(deaths)):
        raise ValueError(
            "The death values contain NaN or infinite values."
        )

    if not np.all(np.isfinite(persistences)):
        raise ValueError(
            "The persistence values contain NaN or infinite values."
        )

    if np.any(persistences < 0):
        raise ValueError(
            "A negative persistence value was found. "
            "Persistence values must be nonnegative."
        )

    max_persistence = float(
        np.max(persistences)
    )

    # When every persistence value is zero, no positive-persistence
    # loop can be selected meaningfully.

    if max_persistence <= 0:
        if dataset_type == "network":
            raise ValueError(
                "The network dataset has no positive-persistence "
                "H1 feature. AdaMapper cannot be run.")

        print("\nAll H1 persistence values are zero.")
        print("Standard Mapper should be used.")
        return [], None, None, "standard_mapper", "run"

    # --------------------------------------------------
    # Slider range
    # --------------------------------------------------
    min_T = 0.0
    if dataset_type == "network":
        max_T = 1.0
    else:
        max_T = 1.10

    if not min_T <= default_T <= max_T:
        raise ValueError(
            f"default_T must be within the slider range "
            f"[{min_T:.2f}, {max_T:.2f}], "
            f"but received {default_T:.4f}."
        )

    slider_start_T = float(
        default_T
    )

    # --------------------------------------------------
    # Mutable callback state
    # --------------------------------------------------
    selected_T = {
        "value": slider_start_T,
    }

    selected_features = {
        "value": [],
    }

    selected_cutoff = {
        "value": slider_start_T * max_persistence,
    }

    algorithm_mode = {
        "value": "adamapper",
    }

    # "pending" means the user has not yet chosen Apply,
    # Default, or Quit.
    threshold_action = {
        "value": "pending",
    }

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------
    fig = plt.figure(
        figsize=(14, 7)
    )

    try:
        fig.canvas.manager.set_window_title(
            f"Persistence Threshold — "
            f"{data_type} — Round {round_number}"
        )
    except AttributeError:
        # Some Matplotlib backends do not implement window titles.
        pass

    ax = fig.add_axes(
        [0.06, 0.16, 0.52, 0.70]
    )

    fig.suptitle(
        "Select Persistence Threshold",
        fontsize=16,
        fontweight="bold",
        y=0.94
    )

    # --------------------------------------------------
    # Persistence-diagram points
    # --------------------------------------------------
    ax.scatter(
        births,
        deaths,
        s=90,
        color="tab:orange",
        alpha=1.0,
        label=r"$H_1$ features",
    )

    min_val = min(
        np.min(births),
        np.min(deaths),
    )

    max_val = max(
        np.max(births),
        np.max(deaths),
    )

    padding = (
        0.05 * (max_val - min_val)
        if max_val > min_val
        else 1.0
    )

    x_min = min_val - padding
    x_max = max_val + padding
    y_min = min_val - padding
    y_max = max_val + padding

    x_line = np.linspace(
        x_min,
        x_max,
        200,
    )

    # --------------------------------------------------
    # Diagonal: death = birth
    # --------------------------------------------------
    ax.plot(
        x_line,
        x_line,
        "k--",
        linewidth=2,
        label=r"$death = birth$",
    )

    # --------------------------------------------------
    # Initial threshold line
    # --------------------------------------------------
    initial_cutoff = (
        slider_start_T * max_persistence
    )

    threshold_line, = ax.plot(
        x_line,
        x_line + initial_cutoff,
        "r--",
        linewidth=3,
        label="Persistence threshold line",
    )

    # --------------------------------------------------
    # Retained-feature highlighting
    # --------------------------------------------------
    selected_scatter = ax.scatter(
        [],
        [],
        s=130,
        facecolors="none",
        edgecolors="green",
        linewidths=2,
        label=r"Retained $H_1$ features (loops)",
    )

    ax.set_xlabel(
        "Birth",
        fontsize=18,
        fontweight="bold",
        labelpad=8,
    )

    ax.set_ylabel(
        "Death",
        fontsize=18,
        fontweight="bold",
        labelpad=8,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    # Hide numerical axis tick labels.
    ax.set_xticks([])
    ax.set_yticks([])

    ax.grid(False)

    ax.legend(
        loc="lower right",
        fontsize=11,
        framealpha=0.9,
    )

    # --------------------------------------------------
    # Right-side explanation box
    # --------------------------------------------------
    fig.text(
        0.63, 0.86,
        "\nT = Persistence threshold \n\n"
        "max_persistence = persistence of "
        "the most persistent $H_1$ feature \n\n"
        "Persistence cutoff = T × max_persistence \n\n"
        "Significant loops / Retained loops: "
        "persistence ≥ persistence cutoff\n",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=1.0
        )
    )

    # --------------------------------------------------
    # Right-side dynamic information box
    # --------------------------------------------------
    count_text = fig.text(
        0.63,
        0.64,
        "",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=1.0,
        ),
    )

    # --------------------------------------------------
    # Right-side status box
    # --------------------------------------------------
    status_text = fig.text(
        0.63,
        0.42,
        "",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=1.0,
        ),
    )

    # --------------------------------------------------
    # Right-side slider
    # --------------------------------------------------
    slider_ax = fig.add_axes(
        [0.64, 0.20, 0.27, 0.04]
    )

    T_slider = Slider(
        ax=slider_ax,
        label="T",
        valmin=min_T,
        valmax=max_T,
        valinit=slider_start_T,
        valfmt="%.4f",
    )

    T_slider.valtext.set_fontweight(
        "bold"
    )

    T_slider.valtext.set_fontsize(
        11
    )

    T_slider.label.set_fontweight(
        "bold"
    )

    T_slider.label.set_fontsize(
        12
    )

    # --------------------------------------------------
    # Right-side buttons
    # --------------------------------------------------

    # Apply the current slider value.
    apply_button_ax = fig.add_axes(
        [0.6, 0.10, 0.13, 0.06]
    )

    apply_button = Button(
        apply_button_ax,
        "Apply selected T",
    )

    apply_button.label.set_fontsize(
        9
    )

    # Restore and apply T=default_T.
    default_button_ax = fig.add_axes(
        [0.74, 0.10, 0.15, 0.06]
    )

    default_button = Button(
        default_button_ax,
        f"Use default T={slider_start_T:.2f}",
    )

    default_button.label.set_fontsize(
        9
    )

    # Quit the entire program.
    quit_button_ax = fig.add_axes(
        [0.9, 0.10, 0.08, 0.06]
    )

    quit_button = Button(
        quit_button_ax,
        "Quit Program",
    )

    quit_button.label.set_fontsize(
        9
    )

    # --------------------------------------------------
    # Update retained-feature highlighting
    # --------------------------------------------------
    def update_selected_scatter(mask):
        """
        Update the green outlines around retained H1 features.
        """

        if np.any(mask):
            retained_points = np.column_stack(
                (
                    births[mask],
                    deaths[mask],
                )
            )
        else:
            # Matplotlib expects shape (0, 2) for an empty scatter.
            retained_points = np.empty(
                (0, 2)
            )

        selected_scatter.set_offsets(
            retained_points
        )

    # --------------------------------------------------
    # Main plot-update callback
    # --------------------------------------------------
    def update_plot(T):
        """
        Update the threshold line, selected features, information
        box, and algorithm mode.
        """

        T = float(T)

        cutoff = (
            T * max_persistence
        )

        selected_T["value"] = T
        selected_cutoff["value"] = cutoff

        # Move the threshold line:
        #
        #     death = birth + cutoff
        threshold_line.set_ydata(
            x_line + cutoff
        )

        # T=0 retains every nonnegative-persistence feature.
        #
        # T>1 places the cutoff above max_persistence and
        # therefore retains no features.
        mask = (
            persistences >= cutoff
        )

        retained_count = int(
            np.sum(mask)
        )

        total_count = len(features)

        selected_features["value"] = [
            feature
            for feature, keep in zip(features, mask)
            if keep
        ]

        update_selected_scatter(
            mask
        )

        # --------------------------------------------------
        # Interpret the current selection
        # --------------------------------------------------
        
        if dataset_type == "network":

            if retained_count == 0:
                raise RuntimeError(
                    "Internal error: a network threshold retained "
                    "zero H1 features."
                )

            if retained_count == total_count:
                status = (
                    "Using AdaMapper\n"
                    "All loops are retained."
                )

            else:
                status = (
                    "Using AdaMapper\n"
                )

            algorithm_mode["value"] = (
                "adamapper"
            )
    
        else:

            if retained_count == 0:
                status = (
                    "Using Standard Mapper\n"
                    "The user does not identify any loop as significant. "
                    
                )

                algorithm_mode["value"] = (
                    "standard_mapper"
                )

            elif retained_count == total_count:
                status = (
                    "Using AdaMapper\n"
                    "All loops are retained."
                )

                algorithm_mode["value"] = (
                    "adamapper"
                )

            else:
                status = (
                    "Using AdaMapper\n"
                )

                algorithm_mode["value"] = (
                    "adamapper"
                )


        count_text.set_text(
            f"\n"
            f" T = $\\mathbf{{{T:.2f}}}$\n\n"
            f" max_persistence = $\\mathbf{{{max_persistence:.4f}}}$\n\n"
            f" Persistence cutoff = $\\mathbf{{{cutoff:.4f}}}$\n\n"
            f" Significant loops / Retained loops = $\\mathbf{{{retained_count}}}$ / {total_count}\n"
        )

        status_text.set_text(
            f"\n"
            f"Status:\n\n"
            f"{status}\n"
        )

        fig.canvas.draw_idle()

    # --------------------------------------------------
    # Save the accepted persistence-selection figure
    # --------------------------------------------------
    def save_selection_figure():
        """
        Save the current threshold-selection figure for this round.
        """
        if output_dir is None:
            raise ValueError("output_dir cannot be None.")
        
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        if output_prefix is None:
            file_prefix = (
                data_type
                .strip()
                .replace(" ", "_"))

        else:
            file_prefix = output_prefix

        output_file = os.path.join(
            output_dir,
            f"{file_prefix}_persistence_selection.png")

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight")

        print(
            f"\nSaved persistence selection: "
            f"{output_file}")
        
    # --------------------------------------------------
    # Slider callback
    # --------------------------------------------------
    def on_slider_change(value):
        update_plot(
            value
        )

    # --------------------------------------------------
    # Apply selected threshold
    # --------------------------------------------------
    def on_apply_button_clicked(event):
        threshold_action["value"] = "run"

        save_selection_figure()

        plt.close(
            fig
        )

    # --------------------------------------------------
    # Apply default threshold
    # --------------------------------------------------
    def on_default_button_clicked(event):
        # set_val() calls update_plot() through the slider callback.
        T_slider.set_val(
            slider_start_T
        )

        threshold_action["value"] = "run"

        save_selection_figure()

        plt.close(
            fig
        )

    # --------------------------------------------------
    # Quit Program
    # --------------------------------------------------
    def on_quit_button_clicked(event):
        threshold_action["value"] = "quit"

        plt.close(
            fig
        )

    # --------------------------------------------------
    # Handle closing the window using X
    # --------------------------------------------------
    def on_figure_closed(event):
        # plt.close(fig) also generates a close_event.
        #
        # Therefore, change the action to quit only if no button
        # has already selected "run" or "quit".
        if threshold_action["value"] == "pending":
            threshold_action["value"] = "quit"

    # --------------------------------------------------
    # Connect callbacks
    # --------------------------------------------------
    T_slider.on_changed(
        on_slider_change
    )

    apply_button.on_clicked(
        on_apply_button_clicked
    )

    default_button.on_clicked(
        on_default_button_clicked
    )

    quit_button.on_clicked(
        on_quit_button_clicked
    )

    fig.canvas.mpl_connect(
        "close_event",
        on_figure_closed,
    )

    # Initialize the selected features and information boxes.
    update_plot(
        slider_start_T
    )

    # Block execution until the figure closes.
    plt.show()

    # Defensive fallback in case a backend closes without emitting
    # a close event.
    if threshold_action["value"] == "pending":
        threshold_action["value"] = "quit"

    return (
        selected_features["value"],
        selected_T["value"],
        selected_cutoff["value"],
        algorithm_mode["value"],
        threshold_action["value"],
    )



def select_mapper_parameters_interactively(
    data_type,
    round_number,
    n_samples,
    current_parameters,
    default_parameters,
    available_filter_functions,
    available_base_point_methods,
    previous_mapper_specific_parameters=None,
):
    """
    Open an interactive window for selecting all parameters required
    by Standard Mapper.

    Parameter-memory behavior
    -------------------------
    This window remembers only parameters previously used by
    Standard Mapper.

    Shared Standard-Mapper parameters are supplied through:

        current_parameters

    Mapper-specific parameters are supplied through:

        previous_mapper_specific_parameters

    Parameters selected in this window
    ----------------------------------
    Always selected:
        filter_function
        overlap_perc
        BP
        n_neighbors
        nr_cubes
        auto_tuning

    When auto_tuning == "off":
        eps
        min_samples

    When auto_tuning == "on":
        eps and min_samples are calculated automatically inside
        Mapper for each cover interval/cube.

        Previous manual eps/min_samples values are preserved so
        that they can reappear if the user later switches
        auto-tuning back to "off".

    Returns
    -------
    dict

        When the user selects Run:

            {
                "action": "run",

                "parameters": {
                    "filter_function": str,
                    "overlap_perc": float,
                    "BP": str or None,
                    "n_neighbors": int,
                    "min_samples": int,
                },

                "nr_cubes": int,
                "eps": float or None,
                "auto_tuning": "on" or "off",
            }

        When auto_tuning is "on":

            decision["eps"] == None

        because the actual epsilon is calculated automatically
        for each cube.

        The remembered manual min_samples value remains inside
        decision["parameters"] so it is not lost between Mapper
        rounds.

        When the user quits:

            {
                "action": "quit",
                "parameters": None,
                "nr_cubes": None,
                "eps": None,
                "auto_tuning": None,
            }
    """

    # ==================================================
    # 1. Validate inputs from the main program
    # ==================================================

    if not isinstance(data_type, str) or not data_type.strip():
        raise ValueError(
            "data_type must be a nonempty string."
        )

    if not isinstance(round_number, int) or round_number < 1:
        raise ValueError(
            "round_number must be an integer greater than "
            "or equal to 1."
        )

    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError(
            "n_samples must be an integer greater than "
            "or equal to 2."
        )

    required_parameter_names = {
        "filter_function",
        "overlap_perc",
        "BP",
        "n_neighbors",
        "min_samples",
    }

    missing_current_parameters = (
        required_parameter_names
        - set(current_parameters)
    )

    if missing_current_parameters:
        raise ValueError(
            "current_parameters is missing the following "
            f"value(s): {sorted(missing_current_parameters)}"
        )

    missing_default_parameters = (
        required_parameter_names
        - set(default_parameters)
    )

    if missing_default_parameters:
        raise ValueError(
            "default_parameters is missing the following "
            f"value(s): {sorted(missing_default_parameters)}"
        )

    available_filter_functions = list(
        available_filter_functions
    )

    available_base_point_methods = list(
        available_base_point_methods
    )

    if not available_filter_functions:
        raise ValueError(
            "available_filter_functions cannot be empty."
        )

    if not available_base_point_methods:
        raise ValueError(
            "available_base_point_methods cannot be empty."
        )

    if (
        current_parameters["filter_function"]
        not in available_filter_functions
    ):
        raise ValueError(
            "The current filter function "
            f"{current_parameters['filter_function']!r} "
            "is not present in available_filter_functions."
        )

    if (
        default_parameters["filter_function"]
        not in available_filter_functions
    ):
        raise ValueError(
            "The default filter function "
            f"{default_parameters['filter_function']!r} "
            "is not present in available_filter_functions."
        )

    if (
        current_parameters["BP"]
        not in available_base_point_methods
    ):
        raise ValueError(
            "The current base-point method "
            f"{current_parameters['BP']!r} "
            "is not present in available_base_point_methods."
        )

    if (
        default_parameters["BP"]
        not in available_base_point_methods
    ):
        raise ValueError(
            "The default base-point method "
            f"{default_parameters['BP']!r} "
            "is not present in available_base_point_methods."
        )

    # --------------------------------------------------
    # Mapper-specific previous values
    # --------------------------------------------------

    if previous_mapper_specific_parameters is None:
        previous_mapper_specific_parameters = {
            "nr_cubes": None,
            "eps": None,
            "auto_tuning": None,
        }

    required_mapper_specific_names = {
        "nr_cubes",
        "eps",
        "auto_tuning",
    }

    missing_mapper_specific_parameters = (
        required_mapper_specific_names
        - set(previous_mapper_specific_parameters)
    )

    if missing_mapper_specific_parameters:
        raise ValueError(
            "previous_mapper_specific_parameters is missing "
            "the following value(s): "
            f"{sorted(missing_mapper_specific_parameters)}"
        )

    previous_nr_cubes = (
        previous_mapper_specific_parameters["nr_cubes"]
    )

    previous_eps = (
        previous_mapper_specific_parameters["eps"]
    )

    previous_auto_tuning = (
        previous_mapper_specific_parameters["auto_tuning"]
    )

    if (
        previous_auto_tuning is not None
        and previous_auto_tuning not in {"on", "off"}
    ):
        raise ValueError(
            "Previous Mapper auto_tuning must be "
            "'on', 'off', or None."
        )

    # ==================================================
    # 2. Default decision
    # ==================================================

    decision = {
        "action": "quit",
        "parameters": None,
        "nr_cubes": None,
        "eps": None,
        "auto_tuning": None,
    }

    # ==================================================
    # 3. Create window
    # ==================================================

    root = tk.Tk()

    root.title(
        f"Standard Mapper Parameters — "
        f"{data_type} — Round {round_number}"
    )

    root.resizable(
        False,
        False,
    )

    # ==================================================
    # 4. Variables connected to GUI widgets
    # ==================================================

    # --------------------------------------------------
    # Shared Mapper parameters
    # --------------------------------------------------

    filter_function_var = tk.StringVar(
        value=str(
            current_parameters["filter_function"]
        )
    )

    overlap_perc_var = tk.StringVar(
        value=str(
            current_parameters["overlap_perc"]
        )
    )

    current_bp = current_parameters["BP"]

    base_point_var = tk.StringVar(
        value=(
            "None"
            if current_bp is None
            else str(current_bp)
        )
    )

    n_neighbors_var = tk.StringVar(
        value=str(
            current_parameters["n_neighbors"]
        )
    )

    current_min_samples = (
        current_parameters["min_samples"]
    )

    min_samples_var = tk.StringVar(
        value=(
            ""
            if current_min_samples is None
            else str(current_min_samples)
        )
    )

    # --------------------------------------------------
    # Mapper-specific remembered parameters
    # --------------------------------------------------

    nr_cubes_var = tk.StringVar(
        value=(
            ""
            if previous_nr_cubes is None
            else str(previous_nr_cubes)
        )
    )

    auto_tuning_var = tk.StringVar(
        value=(
            ""
            if previous_auto_tuning is None
            else str(previous_auto_tuning)
        )
    )

    eps_var = tk.StringVar(
        value=(
            ""
            if previous_eps is None
            else str(previous_eps)
        )
    )

    base_point_display_values = [
        "None" if method is None else str(method)
        for method in available_base_point_methods
    ]

    # ==================================================
    # 5. Main container
    # ==================================================

    main_frame = ttk.Frame(
        root,
        padding=(40, 30),
    )

    main_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
    )

    # ==================================================
    # 6. Title
    # ==================================================

    ttk.Label(
        main_frame,
        text="Select Standard Mapper Parameters",
        font=("TkDefaultFont", 15, "bold"),
    ).grid(
        row=0,
        column=0,
        columnspan=3,
        pady=(0, 5),
    )

    ttk.Label(
        main_frame,
        text=(
            f"Dataset: {data_type} — "
            f"Round {round_number}"
        ),
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=1,
        column=0,
        columnspan=3,
        pady=(0, 12),
    )

    # --------------------------------------------------
    # Round-dependent instruction
    # --------------------------------------------------

    if round_number == 1:
        instruction_message = (
            "Select the parameters for Standard Mapper. "
            "The epsilon and minimum-samples fields are required "
            "only when auto-tuning is off."
        )
    else:
        instruction_message = (
            "The epsilon and minimum-samples fields "
            "are required only when auto-tuning is off."
        )

    ttk.Label(
        main_frame,
        text=instruction_message,
        wraplength=650,
        justify="center",
    ).grid(
        row=2,
        column=0,
        columnspan=3,
        pady=(0, 18),
    )

    # ==================================================
    # 7. Table headings
    # ==================================================

    ttk.Label(
        main_frame,
        text="Parameter",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=3,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=(0, 8),
    )

    ttk.Label(
        main_frame,
        text="Value for This Round",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=3,
        column=1,
        sticky="w",
        padx=(0, 20),
        pady=(0, 8),
    )

    ttk.Label(
        main_frame,
        text="Recommended Default",
        font=("TkDefaultFont", 10, "bold"),
    ).grid(
        row=3,
        column=2,
        sticky="w",
        pady=(0, 8),
    )

    # ==================================================
    # Helper for recommended defaults
    # ==================================================

    def create_default_label(row, value):

        display_value = (
            "None"
            if value is None
            else str(value)
        )

        ttk.Label(
            main_frame,
            text=display_value,
        ).grid(
            row=row,
            column=2,
            sticky="w",
            pady=6,
        )

    # ==================================================
    # 8. Filter function
    # ==================================================

    ttk.Label(
        main_frame,
        text="Filter function",
    ).grid(
        row=4,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    filter_function_combobox = ttk.Combobox(
        main_frame,
        textvariable=filter_function_var,
        values=available_filter_functions,
        state="readonly",
        width=34,
    )

    filter_function_combobox.grid(
        row=4,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    create_default_label(
        4,
        default_parameters["filter_function"],
    )

    # ==================================================
    # 9. Overlap percentage
    # ==================================================

    ttk.Label(
        main_frame,
        text="Overlap percentage",
    ).grid(
        row=5,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    overlap_entry = ttk.Entry(
        main_frame,
        textvariable=overlap_perc_var,
        width=37,
    )

    overlap_entry.grid(
        row=5,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    create_default_label(
        5,
        default_parameters["overlap_perc"],
    )

    # ==================================================
    # 10. Base-point method
    # ==================================================

    ttk.Label(
        main_frame,
        text="Base-point method",
    ).grid(
        row=6,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    base_point_combobox = ttk.Combobox(
        main_frame,
        textvariable=base_point_var,
        values=base_point_display_values,
        state="readonly",
        width=34,
    )

    base_point_combobox.grid(
        row=6,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    create_default_label(
        6,
        default_parameters["BP"],
    )

    # ==================================================
    # 11. Number of neighbors
    # ==================================================

    ttk.Label(
        main_frame,
        text="Number of neighbors",
    ).grid(
        row=7,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    n_neighbors_entry = ttk.Entry(
        main_frame,
        textvariable=n_neighbors_var,
        width=37,
    )

    n_neighbors_entry.grid(
        row=7,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    create_default_label(
        7,
        default_parameters["n_neighbors"],
    )

    # ==================================================
    # 12. Number of cubes
    # ==================================================

    ttk.Label(
        main_frame,
        text="Number of cubes",
    ).grid(
        row=8,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    nr_cubes_entry = ttk.Entry(
        main_frame,
        textvariable=nr_cubes_var,
        width=37,
    )

    nr_cubes_entry.grid(
        row=8,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    ttk.Label(
        main_frame,
        text="—",
    ).grid(
        row=8,
        column=2,
        sticky="w",
        pady=6,
    )

    # ==================================================
    # 13. Auto-tuning
    # ==================================================

    ttk.Label(
        main_frame,
        text="Auto-tuning",
    ).grid(
        row=9,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    auto_tuning_combobox = ttk.Combobox(
        main_frame,
        textvariable=auto_tuning_var,
        values=["on", "off"],
        state="readonly",
        width=34,
    )

    auto_tuning_combobox.grid(
        row=9,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    ttk.Label(
        main_frame,
        text="—",
    ).grid(
        row=9,
        column=2,
        sticky="w",
        pady=6,
    )

    # ==================================================
    # 14. Epsilon
    # ==================================================

    eps_label = ttk.Label(
        main_frame,
        text="Clustering epsilon (eps)",
    )

    eps_label.grid(
        row=10,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    eps_entry = ttk.Entry(
        main_frame,
        textvariable=eps_var,
        width=37,
    )

    eps_entry.grid(
        row=10,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    # ==================================================
    # 15. Minimum samples
    # ==================================================

    min_samples_label = ttk.Label(
        main_frame,
        text="Minimum samples",
    )

    min_samples_label.grid(
        row=11,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=6,
    )

    min_samples_entry = ttk.Entry(
        main_frame,
        textvariable=min_samples_var,
        width=37,
    )

    min_samples_entry.grid(
        row=11,
        column=1,
        sticky="ew",
        padx=(0, 20),
        pady=6,
    )

    # ==================================================
    # 16. Dynamically show/hide eps and min_samples
    # ==================================================

    def update_auto_tuning_fields(event=None):

        auto_tuning = (
            auto_tuning_var.get().strip()
        )

        if auto_tuning == "off":

            eps_label.grid()
            eps_entry.grid()

            min_samples_label.grid()
            min_samples_entry.grid()

        else:

            eps_label.grid_remove()
            eps_entry.grid_remove()

            min_samples_label.grid_remove()
            min_samples_entry.grid_remove()

        # --------------------------------------------------
        # Resize the window after showing/hiding fields
        # --------------------------------------------------

        root.update_idletasks()

        window_width = root.winfo_reqwidth() + 80
        window_height = root.winfo_reqheight() + 40

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        window_x = max(
            0,
            (screen_width - window_width) // 2,
        )

        window_y = max(
            0,
            (screen_height - window_height) // 2,
        )

        root.geometry(
            f"{window_width}x{window_height}"
            f"+{window_x}+{window_y}"
        )
        
    auto_tuning_combobox.bind(
        "<<ComboboxSelected>>",
        update_auto_tuning_fields,
    )

    # If this window has been used before, the previous
    # auto-tuning selection determines the initial visibility.
    update_auto_tuning_fields()

    # ==================================================
    # 17. Explanation
    # ==================================================

    note_label = ttk.Label(
        main_frame,
        text=(
            "Auto-tuning ON:\n"
            "Mapper automatically calculates epsilon and minimum "
            "samples separately for each cover interval/cube.\n\n"
            "Auto-tuning OFF:\n"
            "You provide epsilon and minimum samples manually.\n\n"
            "If Standard Mapper was used in an earlier round, "
            "its previous parameters are shown again here.\n\n"
            "Only base_point_geodesic_distance requires a "
            "base-point method. For other filter functions, "
            "select None."
        ),
        wraplength=650,
        justify="left",
    )

    note_label.grid(
        row=12,
        column=0,
        columnspan=3,
        sticky="w",
        pady=(14, 16),
    )

    # ==================================================
    # 18. Button container
    # ==================================================

    button_frame = ttk.Frame(
        main_frame,
    )

    button_frame.grid(
        row=13,
        column=0,
        columnspan=3,
        pady=(4, 0),
    )

    # ==================================================
    # 19. Restore recommended shared parameters
    # ==================================================

    def restore_defaults():
        """
        Restore recommended shared parameters.

        Standard-Mapper-specific parameters do not have
        recommended defaults, so nr_cubes, eps, and auto_tuning
        are cleared.
        """

        filter_function_var.set(
            str(
                default_parameters["filter_function"]
            )
        )

        overlap_perc_var.set(
            str(
                default_parameters["overlap_perc"]
            )
        )

        default_bp = default_parameters["BP"]

        base_point_var.set(
            "None"
            if default_bp is None
            else str(default_bp)
        )

        n_neighbors_var.set(
            str(
                default_parameters["n_neighbors"]
            )
        )

        default_min_samples = (
            default_parameters["min_samples"]
        )

        min_samples_var.set(
            ""
            if default_min_samples is None
            else str(default_min_samples)
        )

        # Mapper-specific settings have no recommended default.
        nr_cubes_var.set("")
        auto_tuning_var.set("")
        eps_var.set("")

        update_auto_tuning_fields()

    # ==================================================
    # 20. Validate and accept parameters
    # ==================================================

    def run_with_parameters():

        # --------------------------------------------------
        # Filter function
        # --------------------------------------------------

        filter_function = (
            filter_function_var.get().strip()
        )

        if (
            filter_function
            not in available_filter_functions
        ):
            messagebox.showerror(
                "Invalid Filter Function",
                "Please select a valid filter function.",
                parent=root,
            )

            filter_function_combobox.focus_set()
            return

        # --------------------------------------------------
        # Base-point method
        # --------------------------------------------------

        base_point_display = (
            base_point_var.get().strip()
        )

        if base_point_display == "None":
            base_point_method = None
        else:
            base_point_method = base_point_display

        if (
            base_point_method
            not in available_base_point_methods
        ):
            messagebox.showerror(
                "Invalid Base-Point Method",
                "Please select a valid base-point method.",
                parent=root,
            )

            base_point_combobox.focus_set()
            return

        if (
            filter_function
            == "base_point_geodesic_distance"
            and base_point_method is None
        ):
            messagebox.showerror(
                "Base Point Required",
                "The base_point_geodesic_distance filter "
                "requires EP, DR, or BC.",
                parent=root,
            )

            base_point_combobox.focus_set()
            return

        if (
            filter_function
            != "base_point_geodesic_distance"
            and base_point_method is not None
        ):
            messagebox.showerror(
                "Base Point Not Required",
                "This filter function does not use a base point. "
                "Please select None.",
                parent=root,
            )

            base_point_combobox.focus_set()
            return

        # --------------------------------------------------
        # Overlap percentage
        # --------------------------------------------------

        try:
            overlap_perc = float(
                overlap_perc_var.get().strip()
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Overlap Percentage",
                "Overlap percentage must be numerical.",
                parent=root,
            )

            overlap_entry.focus_set()
            return

        if not np.isfinite(overlap_perc):

            messagebox.showerror(
                "Invalid Overlap Percentage",
                "Overlap percentage must be finite.",
                parent=root,
            )

            overlap_entry.focus_set()
            return

        if not 0.0 <= overlap_perc < 1.0:

            messagebox.showerror(
                "Invalid Overlap Percentage",
                "Overlap percentage must satisfy "
                "0 ≤ overlap_perc < 1.",
                parent=root,
            )

            overlap_entry.focus_set()
            return

        # --------------------------------------------------
        # Number of neighbors
        # --------------------------------------------------

        try:
            n_neighbors = int(
                n_neighbors_var.get().strip()
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Number of Neighbors",
                "Number of neighbors must be an integer.",
                parent=root,
            )

            n_neighbors_entry.focus_set()
            return

        if n_neighbors < 1:

            messagebox.showerror(
                "Invalid Number of Neighbors",
                "Number of neighbors must be at least 1.",
                parent=root,
            )

            n_neighbors_entry.focus_set()
            return

        if n_neighbors >= n_samples:

            messagebox.showerror(
                "Invalid Number of Neighbors",
                "Number of neighbors must be smaller than "
                "the number of data points.\n\n"
                f"Number of data points: {n_samples}\n"
                f"Entered n_neighbors: {n_neighbors}",
                parent=root,
            )

            n_neighbors_entry.focus_set()
            return

        # --------------------------------------------------
        # Number of cubes
        # --------------------------------------------------

        try:
            nr_cubes = int(
                nr_cubes_var.get().strip()
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Number of Cubes",
                "Number of cubes must be an integer.",
                parent=root,
            )

            nr_cubes_entry.focus_set()
            return

        if nr_cubes < 1:

            messagebox.showerror(
                "Invalid Number of Cubes",
                "Number of cubes must be at least 1.",
                parent=root,
            )

            nr_cubes_entry.focus_set()
            return

        if nr_cubes > n_samples:

            messagebox.showerror(
                "Invalid Number of Cubes",
                "Number of cubes cannot exceed the "
                "number of data points.\n\n"
                f"Number of data points: {n_samples}\n"
                f"Entered nr_cubes: {nr_cubes}",
                parent=root,
            )

            nr_cubes_entry.focus_set()
            return

        # --------------------------------------------------
        # Auto-tuning
        # --------------------------------------------------

        auto_tuning = (
            auto_tuning_var.get().strip()
        )

        if auto_tuning not in {"on", "off"}:

            messagebox.showerror(
                "Invalid Auto-Tuning Selection",
                "Please select either on or off.",
                parent=root,
            )

            auto_tuning_combobox.focus_set()
            return

        if (
            auto_tuning == "on"
            and overlap_perc <= 0.0
        ):

            messagebox.showerror(
                "Invalid Overlap Percentage",
                "Auto-tuning requires an overlap percentage "
                "greater than zero.",
                parent=root,
            )

            overlap_entry.focus_set()
            return

        # --------------------------------------------------
        # Auto-tuning OFF
        # --------------------------------------------------

        if auto_tuning == "off":

            try:
                eps = float(
                    eps_var.get().strip()
                )

            except ValueError:

                messagebox.showerror(
                    "Invalid Epsilon",
                    "Clustering epsilon must be numerical.",
                    parent=root,
                )

                eps_entry.focus_set()
                return

            if not np.isfinite(eps):

                messagebox.showerror(
                    "Invalid Epsilon",
                    "Clustering epsilon must be finite.",
                    parent=root,
                )

                eps_entry.focus_set()
                return

            if eps <= 0:

                messagebox.showerror(
                    "Invalid Epsilon",
                    "Clustering epsilon must be greater than 0.",
                    parent=root,
                )

                eps_entry.focus_set()
                return

            try:
                min_samples = int(
                    min_samples_var.get().strip()
                )

            except ValueError:

                messagebox.showerror(
                    "Invalid Minimum Samples",
                    "Minimum samples must be an integer.",
                    parent=root,
                )

                min_samples_entry.focus_set()
                return

            if min_samples < 1:

                messagebox.showerror(
                    "Invalid Minimum Samples",
                    "Minimum samples must be at least 1.",
                    parent=root,
                )

                min_samples_entry.focus_set()
                return

            # This is the actual manual epsilon used by
            # this Standard Mapper run.
            run_eps = eps

        # --------------------------------------------------
        # Auto-tuning ON
        # --------------------------------------------------

        else:

            # The algorithm calculates the actual epsilon
            # and min_samples separately for every cube.
            run_eps = None

            # Preserve the previous/manual min_samples value
            # so the Mapper window does not forget it.
            #
            # It is NOT used for the auto-tuned clustering.
            if current_min_samples is not None:
                min_samples = int(
                    current_min_samples
                )
            else:
                min_samples = int(
                    default_parameters["min_samples"]
                )

        # --------------------------------------------------
        # Store accepted values
        # --------------------------------------------------

        decision["action"] = "run"

        decision["parameters"] = {
            "filter_function": filter_function,
            "overlap_perc": overlap_perc,
            "BP": base_point_method,
            "n_neighbors": n_neighbors,
            "min_samples": min_samples,
        }

        decision["nr_cubes"] = nr_cubes
        decision["eps"] = run_eps
        decision["auto_tuning"] = auto_tuning

        root.destroy()

    # ==================================================
    # 21. Quit
    # ==================================================

    def quit_program():

        decision["action"] = "quit"
        decision["parameters"] = None
        decision["nr_cubes"] = None
        decision["eps"] = None
        decision["auto_tuning"] = None

        root.destroy()

    # ==================================================
    # 22. Buttons
    # ==================================================

    run_button = ttk.Button(
        button_frame,
        text="Run with These Parameters",
        command=run_with_parameters,
    )

    run_button.grid(
        row=0,
        column=0,
        padx=6,
    )

    defaults_button = ttk.Button(
        button_frame,
        text="Use Recommended Defaults",
        command=restore_defaults,
    )

    defaults_button.grid(
        row=0,
        column=1,
        padx=6,
    )

    quit_button = ttk.Button(
        button_frame,
        text="Quit Program",
        command=quit_program,
    )

    quit_button.grid(
        row=0,
        column=2,
        padx=6,
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        quit_program,
    )

    main_frame.columnconfigure(
        1,
        weight=1,
    )

    # ==================================================
    # 23. Center window
    # ==================================================

    root.update_idletasks()

    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_x = max(
        0,
        (screen_width - window_width) // 2,
    )

    window_y = max(
        0,
        (screen_height - window_height) // 2,
    )

    root.geometry(
        f"{window_width}x{window_height}"
        f"+{window_x}+{window_y}"
    )

    root.lift()
    root.focus_force()

    # ==================================================
    # 24. Wait for user
    # ==================================================

    root.mainloop()

    return decision



def ask_after_round_window(
    data_type,
    round_number,
):
    """
    Ask the user what to do after one experiment round finishes.

    Parameters
    ----------
    data_type : str
        Name of the current dataset.

    round_number : int
        Number of the completed experiment round.

    Returns
    -------
    str
        One of:

            "run_another_round"
            "next_dataset"
            "quit_program"

    Notes
    -----
    Closing the window using the X button is treated as
    "quit_program".
    """

    # --------------------------------------------------
    # Validate arguments received from run_dataset()
    # --------------------------------------------------
    if not isinstance(data_type, str) or not data_type.strip():
        raise ValueError(
            "data_type must be a nonempty string."
        )

    if not isinstance(round_number, int) or round_number < 1:
        raise ValueError(
            "round_number must be an integer greater than or equal to 1."
        )

    # --------------------------------------------------
    # Default action
    # --------------------------------------------------
    # If the user closes the window using X, the program quits.
    decision = {
        "value": "quit_program"
    }

    # --------------------------------------------------
    # Create the window
    # --------------------------------------------------
    root = tk.Tk()

    root.title(
        f"Round Completed — {data_type} — Round {round_number}"
    )

    root.resizable(
        False,
        False,
    )

    # --------------------------------------------------
    # Main container
    # --------------------------------------------------
    main_frame = ttk.Frame(
        root,
        padding=(28, 24),
    )

    main_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
    )

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    title_label = ttk.Label(
        main_frame,
        text="Experiment Round Completed",
        font=("TkDefaultFont", 15, "bold"),
    )

    title_label.grid(
        row=0,
        column=0,
        columnspan=3,
        pady=(0, 8),
    )

    # --------------------------------------------------
    # Dataset and round information
    # --------------------------------------------------
    round_label = ttk.Label(
        main_frame,
        text=(
            f"Dataset: {data_type}\n"
            f"Completed round: {round_number}"
        ),
        font=("TkDefaultFont", 11, "bold"),
        justify="center",
    )

    round_label.grid(
        row=1,
        column=0,
        columnspan=3,
        pady=(0, 16),
    )

    # --------------------------------------------------
    # Instructions
    # --------------------------------------------------
    instruction_label = ttk.Label(
        main_frame,
        text=(
            "Choose what you would like to do next.\n\n"
            "Run Another Round reuses the current dataset's "
            "previously calculated persistence diagram.\n"
            "Continue to Next Dataset finishes this dataset and "
            "moves to the next enabled dataset."
        ),
        wraplength=650,
        justify="center",
    )

    instruction_label.grid(
        row=2,
        column=0,
        columnspan=3,
        pady=(0, 22),
    )

    # --------------------------------------------------
    # Button callbacks
    # --------------------------------------------------
    def run_another_round():
        decision["value"] = "run_another_round"
        root.destroy()

    def continue_to_next_dataset():
        decision["value"] = "next_dataset"
        root.destroy()

    def quit_program():
        decision["value"] = "quit_program"
        root.destroy()

    # --------------------------------------------------
    # Buttons
    # --------------------------------------------------
    rerun_button = ttk.Button(
        main_frame,
        text="Run Another Round for This Dataset",
        command=run_another_round,
    )

    rerun_button.grid(
        row=3,
        column=0,
        padx=6,
        pady=(0, 4),
    )

    next_dataset_button = ttk.Button(
        main_frame,
        text="Continue to Next Dataset",
        command=continue_to_next_dataset,
    )

    next_dataset_button.grid(
        row=3,
        column=1,
        padx=6,
        pady=(0, 4),
    )

    quit_button = ttk.Button(
        main_frame,
        text="Quit Program",
        command=quit_program,
    )

    quit_button.grid(
        row=3,
        column=2,
        padx=6,
        pady=(0, 4),
    )

    # Closing with X behaves like Quit Program.
    root.protocol(
        "WM_DELETE_WINDOW",
        quit_program,
    )

    # No Enter or Escape shortcuts are used.
    # The user must explicitly click one of the buttons.

    # --------------------------------------------------
    # Center the window
    # --------------------------------------------------
    root.update_idletasks()

    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_x = max(
        0,
        (screen_width - window_width) // 2,
    )

    window_y = max(
        0,
        (screen_height - window_height) // 2,
    )

    root.geometry(
        f"{window_width}x{window_height}+{window_x}+{window_y}"
    )

    root.lift()
    root.focus_force()

    # Wait until the user clicks a button or closes the window.
    root.mainloop()

    return decision["value"]

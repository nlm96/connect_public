def load_class_contours(class_plots_path, param1, param2, basename, verbose=0):
    """
    Tries to load the pre-computed 68% and 95% contour data
    from a MontePython 'analyze' run.
    """
    
    # Handle the '100*theta_s' vs '100theta_s' inconsistency
    # and the 'log10' prefix case you mentioned.
    def get_name_variations(p):
        variations = [p]
        if p == "100*theta_s":
            variations.append("100theta_s")
        elif p.startswith("log10"):
            variations.append(p.replace("log10", "log10^"))
        else:
            variations.append(f"log10{p}")
            variations.append(f"log10^{p}")
        return list(set(variations)) # unique names

    # param1 is the parameter for the X-AXIS
    # param2 is the parameter for the Y-AXIS
    param1_vars = get_name_variations(param1) 
    param2_vars = get_name_variations(param2) 

    dat_path = None
    file_is_swapped = False 
    
    # --- NYT: Variabler til at gemme hvilket navn der faktisk blev fundet ---
    found_p1_name = None
    found_p2_name = None

    # --- Find the file ---
    path_ji_found = False
    for p2 in param2_vars: # Y-param
        for p1 in param1_vars: # X-param
            path_ji = os.path.join(class_plots_path, f"{basename}_2d_{p2}-{p1}.dat")
            if os.path.exists(path_ji):
                dat_path = path_ji
                file_is_swapped = False
                path_ji_found = True
                # --- NYT: Gem de fundne navne ---
                found_p1_name = p1
                found_p2_name = p2
                break
        if path_ji_found:
            break

    if not dat_path:
        # Fallback: check for X-Y file
        for p1 in param1_vars:
            for p2 in param2_vars:
                path_ij = os.path.join(class_plots_path, f"{basename}_2d_{p1}-{p2}.dat")
                if os.path.exists(path_ij):
                    dat_path = path_ij
                    file_is_swapped = True
                    # --- NYT: Gem de fundne navne ---
                    found_p1_name = p1
                    found_p2_name = p2
                    break
            if dat_path:
                break
    
    if not dat_path:
        return None  # No file found

    try:
        # --- Parse the file ---
        # x_list_file *always* gets col 0, y_list_file *always* gets col 1
        x95_list_file, y95_list_file = [], []
        x68_list_file, y68_list_file = [], []
        
        with open(dat_path) as f:
            current_list_x, current_list_y = None, None
            current_x, current_y = [], []

            for line in f:
                if line.strip().startswith('# contour for confidence level 0.95'):
                    if current_x: 
                        current_list_x.append(current_x)
                        current_list_y.append(current_y)
                    current_list_x, current_list_y = x95_list_file, y95_list_file
                    current_x, current_y = [], []
                    continue
                elif line.strip().startswith('# contour for confidence level 0.68'):
                    if current_x: 
                        current_list_x.append(current_x)
                        current_list_y.append(current_y)
                    current_list_x, current_list_y = x68_list_file, y68_list_file
                    current_x, current_y = [], []
                    continue
                elif line.strip().startswith('#'):
                    continue

                if not line.strip() and current_list_x is not None:
                    if current_x:
                        current_list_x.append(current_x)
                        current_list_y.append(current_y)
                    current_x, current_y = [], []
                    continue

                if current_list_x is None:
                    continue 

                try:
                    parts = line.split()
                    val_x, val_y = float(parts[0]), float(parts[1])
                    current_x.append(val_x)
                    current_y.append(val_y)
                except (ValueError, IndexError, TypeError):
                    continue 
            
            if current_x and current_list_x is not None:
                current_list_x.append(current_x)
                current_list_y.append(current_y)
                
        
        # --- Assign parameter names to the data we just read ---
        # Også her definerer vi hvilken "fundet" variabel der hører til hvilken kolonne
        if file_is_swapped:
            # File was X-Y.dat. Content assumed (Y, X).
            # So col 0 (x_list_file) is Y-data, col 1 (y_list_file) is X-data.
            x_data_param_name = param2
            y_data_param_name = param1
            
            # Mapped found names to columns
            found_name_col0 = found_p2_name
            found_name_col1 = found_p1_name
        else:
            # File was Y-X.dat. Content is (X, Y).
            # So col 0 (x_list_file) is X-data, col 1 (y_list_file) is Y-data.
            x_data_param_name = param1
            y_data_param_name = param2
            
            # Mapped found names to columns
            found_name_col0 = found_p1_name
            found_name_col1 = found_p2_name

        # --- NYT: Tjek om vi skal konvertere log10 data til lineær ---
        # Tjek Kolonne 0
        if found_name_col0 and found_name_col0.startswith("log10") and not x_data_param_name.startswith("log10"):
            if verbose >= 1:
                print(f"Converting {x_data_param_name} (Col 0) from log10 to linear.")
            x95_list_file = [[10**x for x in sublist] for sublist in x95_list_file]
            x68_list_file = [[10**x for x in sublist] for sublist in x68_list_file]

        # Tjek Kolonne 1
        if found_name_col1 and found_name_col1.startswith("log10") and not y_data_param_name.startswith("log10"):
            if verbose >= 1:
                print(f"Converting {y_data_param_name} (Col 1) from log10 to linear.")
            y95_list_file = [[10**y for y in sublist] for sublist in y95_list_file]
            y68_list_file = [[10**y for y in sublist] for sublist in y68_list_file]
        # -------------------------------------------------------------

        # --- Scale the data lists based on their assigned parameter name ---
        # (Din eksisterende omega_b logik)
        if x_data_param_name == 'omega_b':
            if x95_list_file and x95_list_file[0] and np.mean(x95_list_file[0]) > 1.0:
                if verbose >= 2: 
                    print(f"Rescaling param '{x_data_param_name}' (file col 0) by /100.")
                x95_list_file = [[x / 100.0 for x in sublist] for sublist in x95_list_file]
                x68_list_file = [[x / 100.0 for x in sublist] for sublist in x68_list_file]

        if y_data_param_name == 'omega_b':
            if y95_list_file and y95_list_file[0] and np.mean(y95_list_file[0]) > 1.0:
                if verbose >= 2: 
                    print(f"Rescaling param '{y_data_param_name}' (file col 1) by /100.")
                y95_list_file = [[y / 100.0 for y in sublist] for sublist in y95_list_file]
                y68_list_file = [[y / 100.0 for y in sublist] for sublist in y68_list_file]
        
        # --- Return the data in the requested X, Y order ---
        if file_is_swapped:
            return y95_list_file, x95_list_file, y68_list_file, x68_list_file
        else:
            return x95_list_file, y95_list_file, x68_list_file, y68_list_file
            
    except Exception as e:
        print(f"Warning: Could not parse contour file {dat_path}. Error: {e}")
        return None
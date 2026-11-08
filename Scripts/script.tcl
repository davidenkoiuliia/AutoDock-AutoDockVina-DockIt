mol addfile D:/3C_prepared/step3_input.parm7 type parm7 waitfor all
mol addfile D:/3C_prepared/prod.xtc type xtc waitfor all

# Open a file to save the results
set out_file [open "bsa_results.txt" "w"]

# Define the probe radius (1.4 Å for water)
set probe_radius 1.4

# Loop through all 1000 frames and calculate SASA for complex, protein, and ligand
for {set frame 0} {$frame < 1000} {incr frame} {
    # Set the current frame
    molinfo top set frame $frame

    display update

    # SASA of the entire complex
    set sel_complex [atomselect top "protein or resname 20T"]
    set sasa_complex [measure sasa $probe_radius $sel_complex]
    $sel_complex delete

    # SASA of the protein
    set sel_protein [atomselect top "protein"]
    set sasa_protein [measure sasa $probe_radius $sel_protein]
    $sel_protein delete

    # SASA of the ligand (resname 20T)
    set sel_ligand [atomselect top "resname 20T"]
    set sasa_ligand [measure sasa $probe_radius $sel_ligand]
    $sel_ligand delete

    # Calculate BSA
    set bsa [expr (($sasa_protein + $sasa_ligand)- $sasa_complex) / 2.0]

    # Print and save the results
    puts "Frame $frame: SASA Complex = $sasa_complex, SASA Protein = $sasa_protein, SASA Ligand = $sasa_ligand, BSA = $bsa"
    puts $out_file "Frame $frame: SASA Complex = $sasa_complex, SASA Protein = $sasa_protein, SASA Ligand = $sasa_ligand, BSA = $bsa"
}

# Close the output file
close $out_file


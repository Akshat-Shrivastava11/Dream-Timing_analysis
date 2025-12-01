import uproot

def list_branches(file_path, tree_name="EventTree"):
    """
    Opens a ROOT file and prints branches inside a tree, one by one.
    """
    print(f"\nOpening file:\n  {file_path}\n")

    # Open file
    f = uproot.open(file_path)

    # Show keys
    print("Keys in ROOT file:")
    for k in f.keys():
        print(" •", k)

    # Load tree
    if tree_name not in f:
        print(f"\nTree '{tree_name}' not found.")
        return

    tree = f[tree_name]
    print(f"\nBranches in {tree_name}:")

    # Loop and print one by one
    for br in tree.keys():
        print(" •", br)

    print("\nDone.\n")

file_path = "/lustre/research/hep/cmadrid/HG-DREAM/CERN/ROOT_TimingDAQ/run1500_250928104710_TimingDAQ.root"
list_branches(file_path)

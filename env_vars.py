import os

# Default File Paths

WDIR = os.path.join("D:\\", "dev", "github", "EstimatingInspectionTasks")

DS_ROOT_PATH = os.path.join("D:\\", "evias_expmts")

RESULTS_PATH = os.path.join(WDIR, "results")

GITHUB_CROSS_APPLICATION_RESULTS = ("https://api.github.com/repos/anmarchio/experiments/contents/test"
                                    "/cross_application/manual_hdev")

# General Image Parameters

IMG_SIZE = 128

SHORT_TO_LONG_NAME = {
    "tile_crack": "MVTec_AD_Tile_Crack",
    "toothbrush_small": "MVTec_AD_Toothbrush_Sm",
    "crack_forest": "CrackForest",
    "cable_missing_train": "MVTec_AD_Cable_Missing",
    "MT_Blowhole_train": "MT_Blowhole_train",
    "kos10": "",
    "kos25": "",
    "FabricDefectsAITEX": "FabricDefectsAITEX",
    "carpet": "MVTec_AD_Carpet",
    "leather": "MVTec_AD_Leather",
    "capsule": "MVTec_AD_Capsule",
    "metal_nut": "MVTec_AD_Metal_Nut",
    "pill": "MVTec_AD_Pill_Crack",
    "grid_thread": "MVTec_AD_Grid_Thread",
    "wood_scratch": "MVTec_AD_Wood_Scratch",
    "zipper": "MVTec_AD_Zipper_Rough",
    "bottle_large": "MVTec_AD_Bottle_Broken_Lg",
    "bottle_small": "MVTec_AD_Bottle_Broken_Sm",
    "hazelnut_crack_train": "MVTec_AD_Hazelnut_Crack",
    "screw_scratch_neck_train": "MVTec_AD_Screw_Scratch",
    "transistor_damaged_case_train": "MVTec_AD_Transistor",
    "81.jpg_bright": "AirCarbon3_80.jpg_bright",
    "81.jpg_dark": "AirCarbon3_81.jpg_dark",
    "80.jpg_dark_1": "AirCarbon3_80.jpg_dark_1",
    "80.jpg_dark_2": "AirCarbon3_80.jpg_dark_2",
    "80.jpg_dark_3": "AirCarbon3_80.jpg_dark_3",
    "80.jpg_dark_4": "AirCarbon3_80.jpg_dark_4",
    "80.jpg_dark_5": "AirCarbon3_80.jpg_dark_5",
    "t_8": "AirCarbon2_t_8.jpg",
    "CF_RefSet": "CF_ReferenceSet",
    "CF_RefSet_Small_Light": "CF_ReferenceSet_Small_Light",
    "CF_ReferenceSet_Small_Dark": "CF_ReferenceSet_Small_Dark",
    "resin_cgp": "Pultrusion_Resin",
    "window_cgp": "Pultrusion_Window",
    "resin_cgp_augmented": "Pultrusion_Resin_Augmtd_mean",
    "severstal": "severstal-steel",
    "spule-upside-0315": "MAIPreform2_Spule0_0315_Upside_Thread_256",
    "spule-upside-0117": "",
    "spule2-0816_Upside-dl1": "MAIPreform2_Spule0_0816_Upside",
    "spule2-0816_Upside-dl2": "",
    "spule-upside-hole": "MAIPreform2_Spule0_0315_Upside_Thread",
    "spule-upside-hole256": "MAIPreform2_Spule0_0315_Upside_Thread_256"
}

LONG_TO_SHORT_NAME = {
    "MVTec_AD_Tile_Crack": "tile_crack",
    "MVTec_AD_Toothbrush_Sm": "toothbrush_small",
    "CrackForest": "crack_forest",
    "MVTec_AD_Cable_Missing": "cable_missing_train",
    "MT_Blowhole_train": "MT_Blowhole_train",
    "Kos10": "kos10",
    "Kos25": "kos25",
    "FabricDefectsAITEX": "FabricDefectsAITEX",
    "MVTec_AD_Carpet": "carpet",
    "MVTec_AD_Leather": "leather",
    "MVTec_AD_Capsule": "capsule",
    "MVTec_AD_Metal_Nut": "metal_nut",
    "MVTec_AD_Pill_Crack": "pill",
    "MVTec_AD_Grid_Thread": "grid_thread",
    "MVTec_AD_Wood_Scratch": "wood_scratch",
    "MVTec_AD_Zipper_Rough": "zipper",
    "MVTec_AD_Bottle_Broken_Lg": "bottle_large",
    "MVTec_AD_Bottle_Broken_Sm": "bottle_small",
    "MVTec_AD_Hazelnut_Crack": "hazelnut_crack_train",
    "MVTec_AD_Screw_Scratch": "screw_scratch_neck_train",
    "MVTec_AD_Transistor": "transistor_damaged_case_train",
    "AirCarbon3_80.jpg_bright": "81.jpg_bright",
    "AirCarbon3_81.jpg_dark": "81.jpg_dark",
    "AirCarbon3_80.jpg_dark_1": "80.jpg_dark_1",
    "AirCarbon3_80.jpg_dark_2": "80.jpg_dark_2",
    "AirCarbon3_80.jpg_dark_3": "80.jpg_dark_3",
    "AirCarbon3_80.jpg_dark_4": "80.jpg_dark_4",
    "AirCarbon3_80.jpg_dark_5": "80.jpg_dark_5",
    "AirCarbon2_t_8.jpg": "t_8",
    "CF_ReferenceSet": "CF_RefSet",
    "CF_ReferenceSet_Small_Light": "CF_RefSet_Small_Light",
    "CF_ReferenceSet_Small_Dark": "CF_ReferenceSet_Small_Dark",
    "Pultrusion_Resin": "resin_cgp",
    "Pultrusion_Window": "window_cgp",
    "Pultrusion_Resin_Augmtd_mean": "resin_cgp_augmented",
    "severstal-steel": "severstal",
    "MAIPreform2_Spule0_0315_Upside_Thread_256": "spule-upside-0315",
    "MAIPreform2_Spule_0117": "spule-upside-0117",
    "MAIPreform2_Spule0_0816_Upside": "spule2-0816_Upside-dl1",
    "MAIPreform2_Spule2-0816": "spule2-0816_Upside-dl2",
    "MAIPreform2_Spule0_0315_Upside_Thread": "spule-upside-hole",
    "MAIPreform2_Spule0_0315_Upside_Thread_256": "spule-upside-hole256"
}

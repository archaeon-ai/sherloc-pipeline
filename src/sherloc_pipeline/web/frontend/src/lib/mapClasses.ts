// Domain → class_id mapping: the fitting engine's class ids, one layer per
// class on the map. Shared by MapMode (which builds the layers) and by
// RGBMixPanel's presets, which must name ids from this table.
export const DOMAIN_CLASSES: Record<string, string[]> = {
  minerals: ['olivine', 'phosphate', 'pyroxene', 'sulf1_v1', 'sulf2_v1',
             '1050', 'lo-carb', 'hi-carb', 'sulf_v3'],
  organics: ['D_band', 'G_band'],
  hydration: ['OH_stretch'],
  fluorescence: ['group3', 'group1a', 'group1b', 'group2'],
};

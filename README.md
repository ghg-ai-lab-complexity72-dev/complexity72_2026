# Regional evolution of greenhouse-gas growth rates from multi-satellite observations

A Complexity72h 2026 (Northeastern University, London) project.

Project description: 

Greenhouse-gas (GHG) concentrations have been rising steadily since the industrial revolution, and many countries have pledged to reduce emissions under the Paris Agreement. A key scientific and societal challenge is determining whether such reductions are actually detectable in the atmosphere. This is difficult because natural variability—such as year-to-year changes in biospheric CO₂ uptake—can obscure or mimic emission-driven trends. Disentangling human and natural influences therefore requires analytical approaches capable of handling complex, high-dimensional data.

A wide range of satellite missions now provide global measurements of atmospheric composition, including total-column CO₂ and CH₄, as well as co-emitted pollutants like NO₂ and CO that trace fossil-fuel combustion. Additional products such as solar-induced chlorophyll fluorescence (SIF) offer insight into biospheric activity. These observations are complemented by bottom-up emission inventories and model-based reanalysis datasets, together forming a diverse set of information on the carbon cycle.

This project will integrate these datasets within a machine-learning and complex-systems framework to explore how GHG trends evolve across different regions. Students will investigate questions such as: Where do CO₂ and CH₄ growth rates show unexpected changes? Which patterns are driven by emissions, and which arise from natural variability? Are there emerging anomalies or regional behaviours that standard analyses overlook? This interdisciplinary project is suitable for students from atmospheric science, physics, computer science, or related fields who are interested in working with real-world environmental data and modern analytical tools.

Team : Adrian Gutierrez Arroyo, Darja Cvetkovic, Sofia Vasquez Alferez, 

## Interactive dashboard from the .nc file

You can generate a single HTML page with controls for `data variable` and `year/time` directly from the netCDF file using xarray.

### 1) Build the dashboard

```bash
python build_interactive_dashboard.py \
	--input unified_annual_carbon_dataset_2015_2024.nc \
	--output multi_variable_dashboard.html \
	--max-points 25000
```

### 2) Open it

Open `multi_variable_dashboard.html` in your browser.

### Notes

- The page is self-contained and interactive (no server needed).
- Controls let you switch both variable and year.
- Color is scaled either with normal or log scale.
- The map is rendered with Bokeh and includes country/land outlines from Natural Earth.
- For speed, map points are sampled, but displayed stats are computed from the full-resolution values for each variable/year.



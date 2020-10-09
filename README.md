# EPA Team 3

### Getting Started

Clone this repo on the server where the database is located:
```
git clone https://github.com/dssg/mlpolicylab_fall20_epa3
```
Install all required packages:
```
cd mlpolicylab_fall20_epa3
pip3 install -r requirements.txt
```

### Running the Code

```
python3 main.py
```
To get information about different command line arguments, run `python3 main.py --help`.


### How It Works

Every experiment is specified by a configuration file. 
Detailed documentation about configuration files can be found in [`experiments/README.md`](https://github.com/dssg/mlpolicylab_fall20_epa3/blob/master/experiments/README.md).

Currently the code reads the config file (default to `experiments/test_run.yaml`), creates config for train/test splits, preprocesses data (which can be skipped using flag `--skip_preprocessing`), generates a cohort table, generates the features, imputes the missing values in the features, trains a set of models while performing grid search over provided set of parameter combinations, and tests each model.

The program creates tables in the `cleaned`, `semantic`, and `experiments` schemas of the database. During training and evaluation, it also creates local model dumps and evaluation result tables in a log directory.

### EPA Data Notes

The New York State Department of Environmental Conservation (NYSDEC) is responsible for inspecting hazardous waste-producing facilities to ensure compliance with federal regulations. As there are over 50,000 facilities and inspections are very time consuming, it is important to ensure that inspectors spend their time inspecting the facilities with the highest risk of violating these regulations.

For this project, you will have access to public waste shipment and inspection data, available from both the Federal Environmental Protection Agency (EPA) and NYSDEC. While national data has been provided for data from the EPA, this project focuses only on inspections in the state of New York. The provided data sources include:
- **RCRAInfo**: Contains inspection, violation, and enforcement data related to the Resource Conservation and Recovery Act (RCRA), as well as information about facilities and handlers of hazardous waste. These data are available in the `rcra` schema, and note in particular the inspections and results information found in `rcra.cmecomp3` (details about this table can be found in the data dictionary under "Data Element Dictionary" -> "Reporting Tables" -> "CM&E"). For more information, see:
    - [RCRAInfo Data Summary](https://echo.epa.gov/tools/data-downloads/rcrainfo-download-summary)
    - [General Information about RCRA](https://rcrapublic.epa.gov/rcrainfoweb/action/main-menu/view)
    - [Detailed Data Dictionary](https://rcrainfo.epa.gov/rcrainfo-help/application/publicHelp/index.htm#introduction.htm)
- **ICIS-FE&C**: Federal enforcement and compliance (FE&C) data from the Integrated Compliance Information System (ICIS), available in the `fec` schema. [More information here](https://echo.epa.gov/tools/data-downloads/icis-fec-download-summary).
- **FRS**: Data in the `frs` schema is from the Facility Registry Service (FRS), allowing for linking facilities between ICIS and RCRAInfo datasets. [More information here](https://echo.epa.gov/tools/data-downloads/frs-download-summary).
- **ICIS-Air**: Data in the `air` schema is from the Integrated Compliance Information System for Air. [More information here](https://echo.epa.gov/tools/data-downloads/icis-air-download-summary).
- **ICIS-NPDES**: Data in the `npdes` schema is from the Integrated Compliance Information System National Pollutant Discharge Elimination System (NPDES). [More information here](https://echo.epa.gov/tools/data-downloads/icis-npdes-download-summary).
- **NYSDEC Reports**: The `nysdec_reports` schema includes information from reports filed annually by large quantity hazardous waste generators as well as treatment, storage, and disposal facilities in the state of New York. For more information, see:
    - [General information about the reports](https://www.dec.ny.gov/chemical/57604.html)
    - [Reporting forms](https://www.dec.ny.gov/chemical/57619.html)
- **Manifest Data**: The `manifest` schema contains information about hazardous waste shipments to, from, or within the state of New York. More information:
    - [Data files and overview](http://www.dec.ny.gov/chemical/9098.html)
    - [General information about manifests](http://www.dec.ny.gov/chemical/60805.html)
    - [Hazardous waste codes and designations](https://govt.westlaw.com/nycrr/Document/I4eacc3f8cd1711dda432a117e6e0f345?viewType=FullText&originationContext=documenttoc&transitionType=CategoryPageItem&contextData=(sc.Default))


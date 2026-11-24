#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

# Generation and Transmission Expansion Planning
# IDAES project
# author: Kyle Skolfield
# date: 01/04/2024
# Model available at http://www.optimization-online.org/DB_FILE/2017/08/6162.pdf
import logging
from pathlib import Path
import csv
import pandas as pd
import os
import random

from prescient.simulator.config import PrescientConfig
from prescient.data.providers import gmlc_data_provider

logger = logging.getLogger("gtep.gtep_data")


class ExpansionPlanningData:
    """Standard data storage class for the IDAES GTEP model."""

    def __init__(
        self,
        stages=2,
        num_reps=4,
        num_commit=24,
        num_dispatch=1,
        duration_representative_period=24,
        save_period_structure_file=False,
        period_structure_json_file=None,
    ):
        """Initialize generation & expansion planning data object.

        :param: stages: integer number of investment periods
        :param: num_reps: integer number of representative periods per investment period
        :param: num_commit: integer number of commitment periods per representative period
        :param: num_dispatch: integer number of dispatch periods per commitment period
        :param: duration_representative_period: duration of each representative period
                (in hours)
        :param: save_period_structure_file: (optional) If True, saves the generated
                period structure as a JSON file in the data directory. Default is False.
        :param: period_structure_json_file: (optional) Path to a JSON file in the data
                directory specifying the period structure. Overrides scalar/list arguments
                if provided. Default is None.

        """
        self.stages = stages
        self.num_reps = num_reps
        self.num_commit = num_commit
        self.num_dispatch = num_dispatch
        self.duration_representative_period = duration_representative_period
        self.save_period_structure_file = save_period_structure_file
        self.period_structure_json_file = period_structure_json_file

    def load_prescient(
        self,
        data_path: Path | str,
        representative_dates: list[str] | None = None,
        representative_weights: dict | None = None,
        options_dict: dict | None = None,
        start_date=None,
        apply_weights_to_full_year: bool = True,
        hours_per_year: float = 8760,
    ):
        """
        Loads data structured via Prescient data loader.

        :param data_path:               Folder containing the data to be loaded.
        :param representative_dates:    List of representative dates to include, each in the
                                            format `"YYYY-MM-DD"`. Defaults to `None`, in
                                            which case dates are automatically chosen.
                                            Note:
                                            Change the last date for whatever
                                            extreme day is needed based on
                                            the given run(s).
        :param representative_weights:  Weight for each representative date, in the format
                                            `{date: weight}`. Every representative date must
                                            be an element of `representative_weights.keys()`.
                                            How the weights are interpreted depends on
                                            `apply_weights_to_full_year`. Defaults to `None`,
                                            in which case all representative dates are
                                            given equal weight.
        :param options_dict:            Arguments to be passed to the Prescient data
                                            loader. Defaults None. If `"num_days"` is not
                                            provided, it is set to `365`. If `"ruc_horizon"` is
                                            not provided, it is set to `36`.
        :param start_date:              Optional simulation start date. If not
                                            provided, this is read from
                                            simulation_objects.csv using the Date_From
                                            row and DAY_AHEAD column, if available.
        :param apply_weights_to_full_year: If True (default), `representative_weights` are
                                            treated as relative values only. GTEP scales
                                            them so the representative periods cover one
                                            full year (`hours_per_year` hours), i.e., so
                                            that sum(weight * duration_representative_period)
                                            = hours_per_year. For example, two 24-hour periods with
                                            weights 1 and 1 each get a weight of 182.5;
                                            with weights 2 and 1 they get 243.33 and
                                            121.67. If False, `representative_weights`
                                            are used directly as the number of times each
                                            period occurs per year and are not checked.
                                            If no weights are passed, each period gets a
                                            weight of 1 before any scaling.
        :param hours_per_year:          Number of hours in the full year that the
                                            weights are scaled to when
                                            `apply_weights_to_full_year` is True.
                                            Defaults to 8760. Not used when
                                            `apply_weights_to_full_year` is False.

        :type data_path:                Path | str
        :type representative_dates:     list | None, optional
        :type representative_weights:   dict | None, optional
        :type options_dict:             dict, optional
        :type apply_weights_to_full_year: bool, optional
        :type hours_per_year:           float, optional
        """
        # GTEP currently reads time-series data by commitment-period
        # index (one hourly data point per commitment period), so the
        # number of commitment periods must equal the representative
        # period duration in hours.
        # [TODO: Remove this restriction once time-series data are
        # mapped to commitment/dispatch periods by time offset.]
        if self.period_structure_json_file is None and (
            self.num_commit != self.duration_representative_period
        ):
            raise ValueError(
                "The number of commitment periods (num_commit="
                f"{self.num_commit}) must equal the representative period "
                "duration in hours (duration_representative_period="
                f"{self.duration_representative_period}). GTEP currently "
                "requires one-hour commitment periods because time-series "
                "data are indexed by commitment period."
            )
        if self.num_dispatch != 1: 
            raise ValueError(
                "GTEP currently only supports the number of dispatch periods "
                "being equal to the number of commitment periods. Any value "
                ">1 results in meaningless dispatch periods."
            )
        self.data_type = "prescient"

        # Create prescient config object with defaults
        prescient_options = PrescientConfig()

        # If start_date is not provided, read it from
        # simulation_objects.csv using Date_From and DAY_AHEAD. Also,
        # verify that the start date year matches the DAY_AHEAD
        # time-series data.
        if start_date is None:
            start_date = self.get_start_date_from_simulation_objects(data_path)
        self.assert_start_date_matches_day_ahead_files(data_path, start_date)

        # set up options dictionary with default values
        if options_dict is None:
            options_dict = {}

        # Set basic configurations that do not match prescient
        # defaults
        default_options_dict = {
            "num_days": 365,
            "ruc_horizon": 36,
            "data_path": data_path,
            "start_date": start_date,
        }
        for k, v in default_options_dict.items():
            if k not in options_dict:
                options_dict[k] = v

        # Update configuration values based on options dictionary
        prescient_options.set_value(options_dict)

        # Use prescient data provider to load in sequential data for
        # representative periods
        data_list = []

        # Validate required input columns before calling Prescient to
        # provide clearer error messages to avoid parser failures.
        self.validate_required_columns(data_path, "gen.csv")
        self.validate_required_columns(data_path, "bus.csv")
        self.validate_required_columns(data_path, "branch.csv")

        data_provider = gmlc_data_provider.GmlcDataProvider(options=prescient_options)

        # Grab details from simulation objects file (data provider
        # above throws error if no simulation_objects.csv exists)
        metadata_path = os.path.join(data_path, "simulation_objects.csv")
        metadata_df = pd.read_csv(metadata_path, index_col=0)

        # GTEP currently assumes hourly input data. Warn if either
        # time series in simulation_objects.csv has a different
        # resolution (in seconds).
        for sim_type in ("DAY_AHEAD", "REAL_TIME"):
            resolution_sec = int(metadata_df.loc["Period_Resolution"][sim_type])
            if resolution_sec != 3600:
                logger.warning(
                    f"Period_Resolution for {sim_type} in simulation_objects.csv "
                    f"is {resolution_sec} seconds, but GTEP currently assumes "
                    "hourly (3600 second) data. Time-series values may not "
                    "line up with commitment periods as expected."
                )

        # Save to variable for easy calling. Note that Prescient
        # loads the REAL_TIME (actuals) series at this frequency.
        sced_freq_min = prescient_options.sced_frequency_minutes
        if sced_freq_min != 60:
            logger.warning(
                f"sced_frequency_minutes is {sced_freq_min}, but GTEP currently "
                "assumes hourly (60 minute) data steps."
            )

        # Number of data steps to load and the number of steps in
        # each representative period. These come from num_days and
        # duration_representative_period, not from
        # simulation_objects.csv.
        total_num_steps = int(prescient_options.num_days * 24 * 60 / sced_freq_min)
        steps_per_rep = int(self.duration_representative_period * 60 / sced_freq_min)

        # Populate an egret model data with the basic stuff
        self.md = data_provider.get_initial_actuals_model(
            options=prescient_options,
            num_time_steps=total_num_steps,
            minutes_per_timestep=sced_freq_min,
        )

        # fill in renewable actuals
        data_provider.populate_with_actuals(
            options=prescient_options,
            num_time_periods=total_num_steps,
            time_period_length_minutes=sced_freq_min,
            start_time=data_provider._start_time,
            model=self.md,
        )

        self.load_default_data_settings()

        self.load_storage_csv(data_path)

        for gen in self.md.data["elements"]["generator"]:
            if "-c" in gen:  # key/Gen UID in csv file; -c = candidate?
                self.md.data["elements"]["generator"][gen]["in_service"] = False

        # JSC addn
        for branch in self.md.data["elements"]["branch"]:
            if "-c" in branch:  # key/Branch UID
                self.md.data["elements"]["branch"][branch]["in_service"] = False
                self.md.data["elements"]["branch"][branch]["candidate"] = True
            else:
                self.md.data["elements"]["branch"][branch]["candidate"] = False

        for stor in self.md.data["elements"]["storage"]:
            if "-c" in stor:  # key/Branch UID
                self.md.data["elements"]["storage"][stor]["in_service"] = False

        self.load_branch_csv(data_path)

        # Get the timestamps in the loaded data. Representative
        # periods must start at one of these timestamps.
        time_keys = self.md.data["system"]["time_keys"]
        time_key_index = {key: idx for idx, key in enumerate(time_keys)}

        def period_fits_in_data(date):
            """Return True if a representative period of
            duration_representative_period starting at date lies fully
            inside the loaded data."""
            return time_key_index[date] + steps_per_rep <= len(time_keys)

        if representative_dates is None:
            # Pick 4 default representative dates in the year of the
            # loaded data: winter, spring, summer, and fall. If
            # self.num_reps < 4, use only the first self.num_reps
            # default dates. If more than 4 representative periods are
            # requested, keep these 4 defaults and randomly select the
            # remaining dates from the midnight timestamps in the data.
            data_year = pd.to_datetime(time_keys[0]).year
            default_representative_dates = [
                f"{data_year}-01-28 00:00",
                f"{data_year}-04-23 00:00",
                f"{data_year}-07-05 00:00",
                f"{data_year}-10-14 00:00",
            ]

            missing_defaults = [
                date
                for date in default_representative_dates
                if date not in time_key_index
            ]
            if missing_defaults:
                raise ValueError(
                    "The following default representative dates are not in the "
                    f"loaded input data: {missing_defaults}. Please provide a "
                    "custom list of representative_dates."
                )

            if self.num_reps <= len(default_representative_dates):
                representative_dates = default_representative_dates[: self.num_reps]
            else:
                remaining_dates = [
                    date
                    for date in time_keys
                    if date.endswith(" 00:00")
                    and date not in default_representative_dates
                    and period_fits_in_data(date)
                ]
                num_additional = self.num_reps - len(default_representative_dates)
                if len(remaining_dates) < num_additional:
                    raise ValueError(
                        "Not enough available day-start timestamps to select default "
                        "representative dates. Please provide a custom list of "
                        "representative_dates in the driver or reduce num_reps."
                    )

                random_seed = 42
                rng = random.Random(random_seed)
                additional_dates = rng.sample(remaining_dates, num_additional)
                representative_dates = sorted(
                    default_representative_dates + additional_dates,
                    key=lambda date: time_key_index[date],
                )
        else:
            # Validate that the user-provided representative_dates
            # match the requested number of representative periods.
            if len(representative_dates) != self.num_reps:
                raise ValueError(
                    f"The number of provided representative_dates must match num_reps. "
                    f"Received len(representative_dates)={len(representative_dates)}, "
                    f"but num_reps={self.num_reps}."
                )

            # Validate that all user-provided representative dates
            # exist in the loaded timestamps.
            missing_dates = [
                date for date in representative_dates if date not in time_key_index
            ]

            if missing_dates:
                raise ValueError(
                    "The following representative_dates are not valid timestamps in the "
                    f"loaded input data: {missing_dates}"
                )

        # Validate that every representative period fits inside the
        # loaded data.
        dates_past_end = [
            date for date in representative_dates if not period_fits_in_data(date)
        ]
        if dates_past_end:
            raise ValueError(
                "The following representative periods run past the end of the "
                f"loaded input data ({time_keys[-1]}) for "
                f"duration_representative_period={self.duration_representative_period} "
                f"hours: {dates_past_end}"
            )

        self.representative_dates = representative_dates

        if representative_weights:

            if len(representative_dates) != len(representative_weights):
                raise ValueError(
                    (
                        f"Number of representative dates ({len(representative_dates)})"
                        + f" and representative_weights ({len(representative_weights)})"
                        + " must match."
                    )
                )

            missing_dates = [
                date
                for date in representative_dates
                if date not in representative_weights
            ]
            if missing_dates:
                raise ValueError(
                    (
                        "Every representative date must be a key in representative_weights,"
                        f" but {missing_dates} are missing."
                    )
                )

            logger.info(
                (
                    "representative_dates and representative_weights are aligned. "
                    + "Continue building the data modeling object..."
                )
            )

            input_weights = {
                date: representative_weights[date] for date in representative_dates
            }

        else:
            # Default to equal weights for each representative period.
            input_weights = {date: 1 for date in representative_dates}

        if apply_weights_to_full_year:
            # Treat the input weights as relative values and scale
            # them so that the representative periods cover one full
            # year: sum(weight * duration) = hours_per_year.
            if hours_per_year <= 0:
                raise ValueError(
                    f"hours_per_year must be positive, but got {hours_per_year}"
                )

            nonpositive = {
                date: weight for date, weight in input_weights.items() if weight <= 0
            }
            if nonpositive:
                raise ValueError(
                    "representative_weights must be positive when "
                    f"apply_weights_to_full_year=True, but got {nonpositive}"
                )

            hours_represented = (
                sum(input_weights.values()) * self.duration_representative_period
            )
            scaling_factor = hours_per_year / hours_represented
            self.representative_weights_dict = {
                date: weight * scaling_factor for date, weight in input_weights.items()
            }

            logger.info(
                "Scaled representative weights to cover a full year "
                f"({hours_per_year} hours): {self.representative_weights_dict}"
            )
        else:
            # Use the input weights directly as the number of times
            # each representative period occurs per year.
            self.representative_weights_dict = input_weights

        # Read average heat rates from the "HR_avg_0" column in
        # gen.csv and assign them to each generator in self.md. This
        # is done manually because the generator data loaded into
        # self.md does not include heat rate values by default. Units
        # should be in MMBTU/MWh.
        # Also load in the p Max value as "nameplate" so that it can
        # be loaded into renewables; "p_max" itself for renewables
        # gets set by DAY_AHEAD_renewables.csv as a time series
        gen_csv_file = os.path.join(data_path, "gen.csv")
        heat_rate_dict = {}
        nameplate_dict = {}
        with open(gen_csv_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gen_uid = row.get("GEN UID")
                heat_rate_str = row.get("HR_avg_0")
                nameplate_str = row.get("PMax MW")

                if heat_rate_str not in (None, "", "NA"):
                    heat_rate = float(heat_rate_str)
                else:
                    heat_rate = None

                if nameplate_str not in (None, "", "NA"):
                    nameplate = float(nameplate_str)
                    if nameplate == 0:
                        raise ValueError(
                            f"Generator {gen_uid} has a PMax MW of 0. This value must be strictly positive"
                        )
                else:
                    raise ValueError(
                        f"Generator {gen_uid} does not have a valid value entered for `PMax MW`"
                    )

                if gen_uid and heat_rate is not None:
                    heat_rate_dict[gen_uid] = heat_rate

                nameplate_dict[gen_uid] = nameplate
        for gen in self.md.data["elements"]["generator"]:
            if gen in heat_rate_dict:
                self.md.data["elements"]["generator"][gen]["heat_rate"] = (
                    heat_rate_dict[gen]
                )
            else:
                self.md.data["elements"]["generator"][gen]["heat_rate"] = 0

            if gen in nameplate_dict:
                self.md.data["elements"]["generator"][gen]["nameplate"] = (
                    nameplate_dict[gen]
                )
            else:
                raise ValueError(
                    f"Generator {gen} was not loaded into GTEP properly. Doublecheck `gen.csv`"
                )

        thermal_heat_rates = [
            self.md.data["elements"]["generator"][gen].get("heat_rate", 0)
            for gen in self.md.data["elements"]["generator"]
            if self.md.data["elements"]["generator"][gen].get("generator_type")
            == "thermal"
        ]

        if thermal_heat_rates and all(hr == 0 for hr in thermal_heat_rates):
            logger.info(
                "All thermal generators have heat_rate values equal to 0. "
                "Please re-check the input data. Fuel costs are multiplied by "
                "heat_rate, so resulting fuel costs will all be 0."
            )

        # IMPORTANT TO READ: Always add or modify any new elements in
        # self.md.data (such as new time series or parameters) BEFORE
        # creating representative_data using clone_at_time_keys. This
        # ensures all representative ModelData objects will have the
        # new elements.
        time_keys = self.md.data["system"]["time_keys"]
        for date in self.representative_dates:
            key_idx = time_keys.index(date)
            time_key_set = time_keys[key_idx : key_idx + steps_per_rep]
            data_list.append(self.md.clone_at_time_keys(time_key_set))

        self.representative_data = data_list

    def import_load_scaling(self, load_file_name, forecast_years=None):
        """Imports load scaling data for forecast years.

        :param load_file_name: filepath for adjusted forecast excel file
        :param forecast_years: list of years to forecast, defaults to [2025, 2030, 2035]

        """
        adjusted_forecast = pd.read_excel(load_file_name)

        if forecast_years is None:
            forecast_years = [2025, 2030, 2035]

        # check years are valid
        if len(forecast_years) < self.stages:
            raise ValueError(
                "Not enough forecast years for the number of stages of investment"
            )
        elif any(year < 2020 or year > 2050 for year in forecast_years):
            raise ValueError(
                "The list of years includes a year before 2020 or after 2050."
            )

        adjusted_forecast_by_period = adjusted_forecast[
            adjusted_forecast["year"].isin(forecast_years)
        ].copy()

        base_zones = [
            "base_economic_coast",
            "base_economic_east",
            "base_economic_fwest",
            "base_economic_ncent",
            "base_economic_north",
            "base_economic_scent",
            "base_economic_south",
            "base_economic_west",
        ]
        scaled_zones = [
            "coast_net",
            "east_net",
            "fwest_net",
            "ncent_net",
            "north_net",
            "scent_net",
            "south_net",
            "west_net",
        ]
        # zones = ["coast", "east", "fwest", "ncent", "north", "scent", "south", "west"]
        # cap_zones = [zone.upper() for zone in zones]
        zones = ["1", "2", "3", "4", "5", "6", "7", "8"]
        cap_zones = ["1", "2", "3", "4", "5", "6", "7", "8"]
        for i, zone in enumerate(zones):
            adjusted_forecast_by_period["scaled_" + zone] = (
                adjusted_forecast_by_period[scaled_zones[i]]
                / adjusted_forecast_by_period[base_zones[i]]
            )
        column_list = [
            "year",
            "month",
            "day",
            "hour",
            "scaled_1",
            "scaled_2",
            "scaled_3",
            "scaled_4",
            "scaled_5",
            "scaled_6",
            "scaled_7",
            "scaled_8",
        ]
        load_scaling_df = adjusted_forecast_by_period[column_list]
        scaled_names = ["scaled_" + zone for zone in zones]
        name_conversion_dict = dict(zip(scaled_names, cap_zones))
        load_scaling_df = load_scaling_df.rename(columns=name_conversion_dict)
        self.load_scaling = load_scaling_df

    def import_outage_data(self, load_file_name, save_mapped_outage_csv=False):
        """This method imports outage data and maps high-probability
        outages to buses.

        :param load_file_name:         Path to the outage data CSV file.
        :param save_mapped_outage_csv: If True, save the mapped
                                       outage-to-bus records to a CSV
                                       file for inspection.  Defaults
                                       to False.

        """

        # Read the outage probability data from the input CSV file.
        outage_list = pd.read_csv(load_file_name)

        # Select the percentile threshold used to identify
        # high-probability outage events and calculate the outage
        # probability value at that percentile.
        percentile_threshold = 0.9
        threshold_value = outage_list["case_4b_prob"].quantile(percentile_threshold)

        # Keep outage events with probability greater than or equal to
        # the threshold and extract the hour from the timestamp
        # string. Keep only the county FIPS code and extracted hour.
        filtered_outages = outage_list[
            outage_list["case_4b_prob"] >= threshold_value
        ].copy()
        filtered_outages["hour"] = filtered_outages["lim_timestamp"].str.extract(
            r" (\d+):"
        )
        filtered_outages = filtered_outages[["fips_code", "hour"]]

        # Use the directory containing the outage file as the base
        # directory and define paths to the county-to-FIPS and
        # bus-to-county mapping files.
        base_dir = Path(load_file_name).parent
        county_fips_path = base_dir / "county_fips_match.csv"
        bus_to_county_path = base_dir / "Bus_data_gen_weights_mappings.csv"
        county_to_fips = pd.read_csv(county_fips_path)
        bus_to_county = pd.read_csv(bus_to_county_path)

        # Keep the columns needed for mapping counties to FIPS codes
        # and buses to counties. Also, add FIPS codes to the
        # bus-to-county mapping.
        county_to_fips = county_to_fips[["County", "FIPS"]]
        bus_to_county = bus_to_county[["Bus Number", "County"]]
        bus_to_county = bus_to_county.merge(county_to_fips, how="inner", on="County")

        # Map outage FIPS codes to buses using the FIPS code and
        # remove outage records that could not be mapped to a bus.
        bus_hours = pd.merge(
            filtered_outages,
            bus_to_county,
            left_on="fips_code",
            right_on="FIPS",
            how="left",
        )
        bus_hours = bus_hours[bus_hours["Bus Number"].notna()]

        # Optionally save the mapped outage records to a new CSV file.
        if save_mapped_outage_csv:
            csv_path = base_dir / "mapped_outage_bus_hours.csv"
            bus_hours.to_csv(csv_path, index=False)

        # Store the hour and bus number columns and convert them to
        # integers.
        self.bus_hours = bus_hours[["hour", "Bus Number"]]
        self.bus_hours = self.bus_hours.astype(int)

    def load_default_data_settings(self):
        """This method fills in required data fields that are not
        specified in the inputs.

        Many of these values are currently hard-coded because they are
        not set elsewhere in the workflow.

        """

        if "elements" in self.md.data.keys():
            if "generator" in self.md.data["elements"].keys():
                for gen in self.md.data["elements"]["generator"]:

                    # Set lifetime value to default first
                    self.md.data["elements"]["generator"][gen]["lifetime"] = 3

                    if "fuel" in self.md.data["elements"]["generator"][gen].keys():
                        if self.md.data["elements"]["generator"][gen]["fuel"] == "C":
                            if (
                                self.md.data["elements"]["generator"][gen]["in_service"]
                                == False
                            ):
                                self.md.data["elements"]["generator"][gen][
                                    "lifetime"
                                ] = 1
                            else:
                                self.md.data["elements"]["generator"][gen][
                                    "lifetime"
                                ] = 2

                    self.md.data["elements"]["generator"][gen][
                        "spinning_reserve_frac"
                    ] = 0.1
                    self.md.data["elements"]["generator"][gen][
                        "quickstart_reserve_frac"
                    ] = 0.1
                    self.md.data["elements"]["generator"][gen]["capital_multiplier"] = 1
                    self.md.data["elements"]["generator"][gen][
                        "extension_multiplier"
                    ] = 0
                    self.md.data["elements"]["generator"][gen][
                        "max_operating_reserve"
                    ] = 1
                    self.md.data["elements"]["generator"][gen][
                        "max_spinning_reserve"
                    ] = 1
                    self.md.data["elements"]["generator"][gen][
                        "max_quickstart_reserve"
                    ] = 1

                    # Use values from the "Ramp Rate MW/Min" input
                    # column when available. Prescient stores these
                    # values in the generator dictionary as "ramp_q"
                    # and "ramp_agc". If these values are missing, use
                    # 1 MW/min as the default ramp-up and ramp-down
                    # rate.
                    gen_data = self.md.data["elements"]["generator"][gen]
                    if (
                        "ramp_agc" in gen_data
                        and gen_data["ramp_agc"] is not None
                        and str(gen_data["ramp_agc"]).strip() != ""
                        and str(gen_data["ramp_agc"]).strip().upper()
                        not in ["NA", "NAN", "NONE"]
                    ):
                        ramp_rate_value = gen_data["ramp_agc"]

                    elif (
                        "ramp_q" in gen_data
                        and gen_data["ramp_q"] is not None
                        and str(gen_data["ramp_q"]).strip() != ""
                        and str(gen_data["ramp_q"]).strip().upper()
                        not in ["NA", "NAN", "NONE"]
                    ):
                        ramp_rate_value = gen_data["ramp_q"]

                    else:
                        ramp_rate_value = 1

                    self.md.data["elements"]["generator"][gen][
                        "ramp_up_rate"
                    ] = ramp_rate_value
                    self.md.data["elements"]["generator"][gen][
                        "ramp_down_rate"
                    ] = ramp_rate_value

                    self.md.data["elements"]["generator"][gen]["emissions_factor"] = 1
                    self.md.data["elements"]["generator"][gen]["start_fuel"] = 1
                    self.md.data["elements"]["generator"][gen]["investment_cost"] = 1
                    self.md.data["elements"]["generator"][gen].setdefault(
                        "non_fuel_startup_cost", 0
                    )

            if "branch" in self.md.data["elements"].keys():
                for branch in self.md.data["elements"]["branch"]:
                    self.md.data["elements"]["branch"][branch]["loss_rate"] = 0
                    self.md.data["elements"]["branch"][branch]["distance"] = 1
                    self.md.data["elements"]["branch"][branch][
                        "capital_cost"
                    ] = 10000000

        if "system" in self.md.data.keys():
            self.md.data["system"]["min_operating_reserve"] = 0.1
            self.md.data["system"]["min_spinning_reserve"] = 0.1

    def load_storage_csv(self, data_path):
        """This method imports storage data.

        :param data_path: filepath for storage data csv file
        """
        data_path = Path(data_path)

        bus_path = data_path / "bus.csv"
        bus_id_to_name = pd.read_csv(bus_path).set_index("Bus ID")["Bus Name"].to_dict()

        try:
            storage_path = data_path / "storage.csv"
            storage_df = pd.read_csv(storage_path)

            storage_data = {}
            for _, row in storage_df.iterrows():
                name = row["name"]
                storage_data[name] = row.drop("name").to_dict()
                storage_data[name]["bus"] = bus_id_to_name[
                    storage_data[name]["bus"]
                ]  # to match egret behavior

            self.md.data["elements"]["storage"] = storage_data
        except FileNotFoundError:
            print(
                f"Warning: The file '{storage_path}' does not exist. Skipping loading storage data."
            )
            self.md.data["elements"]["storage"] = {}

    def load_branch_csv(self, data_path):
        """This method imports branch data.

        :param data_path: filepath for branch data csv file
        """
        data_path = Path(data_path)

        branch_path = data_path / "branch.csv"
        branch_df = pd.read_csv(branch_path)

        branch_data = {}
        if "Static Line" not in branch_df.columns:
            branch_df["Static Line"] = 0

        for _, row in branch_df.iterrows():
            name = row["UID"]

            value = row["Static Line"]

            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                static_line = False
            elif value in (0, 1):  # Accepts numeric 0/1
                static_line = bool(value)
            else:
                raise ValueError(f"Invalid Static Line value for {name}: {value!r}")

            static_status = (
                static_line and self.md.data["elements"]["branch"][name]["in_service"]
            )

            self.md.data["elements"]["branch"][name]["static"] = static_status

    def texas_case_study_updates(self, data_path):
        """Imports generator data for texas case study.

        :param data_path: filepath for generator data csv file
        """

        # Check that datapath is coming from a texas case study
        # directory
        if (
            ("Texas" not in str(data_path))
            and ("Coal" not in str(data_path))
            and ("Resil_Week" not in str(data_path))
        ):
            raise ValueError("The data path provided is not a Texas case study")

        # Enforce pathlib object
        if not isinstance(data_path, Path):
            data_path = Path(data_path)

        generator_update_path = data_path / "gen.csv"
        generator_df = pd.read_csv(generator_update_path)
        bonus_feature_list = [
            "capex1",
            "capex2",
            "capex3",
            "fuel_cost1",
            "fuel_cost2",
            "fuel_cost3",
            "fixed_ops1",
            "fixed_ops2",
            "fixed_ops3",
            "var_ops1",
            "var_ops2",
            "var_ops3",
        ]
        for data_point in self.representative_data:
            for col in bonus_feature_list:
                for gen in data_point.data["elements"]["generator"]:
                    if not data_point.data["elements"]["generator"][gen].get(col):
                        matching_rows = generator_df[generator_df["GEN UID"] == gen]
                        if not matching_rows.empty:
                            data_point.data["elements"]["generator"][gen][col] = float(
                                matching_rows[col].iloc[0]
                            )

    def get_start_date_from_simulation_objects(self, data_path):
        """This method reads the start date from the
        simulation_objects.csv.

        """

        simulation_objects_file = os.path.join(data_path, "simulation_objects.csv")

        if not os.path.exists(simulation_objects_file):
            return None

        simulation_objects_df = pd.read_csv(simulation_objects_file)

        date_from_row = simulation_objects_df[
            simulation_objects_df["Simulation_Parameters"] == "Date_From"
        ]

        if date_from_row.empty:
            return None

        return str(date_from_row["DAY_AHEAD"].iloc[0]).split()[0]

    def assert_start_date_matches_day_ahead_files(self, data_path, start_date):
        """This method checks that the year in start_date matches
        DAY_AHEAD time-series files.

        """

        if start_date is None:
            return

        start_year = pd.to_datetime(start_date).year

        day_ahead_files = [
            "DAY_AHEAD_load.csv",
            "DAY_AHEAD_renewables.csv",
        ]

        for filename in day_ahead_files:
            file_path = os.path.join(data_path, filename)

            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path, usecols=["Year"])
            file_year = int(df["Year"].dropna().iloc[0])

            # Handle possible two-digit years, example 19 to
            # convert to 2019
            if file_year < 100:
                file_year += 2000

            assert file_year == start_year, (
                f"Start date year ({start_year}) does not match the "
                f"first year in {filename} ({file_year}). Please check "
                "simulation_objects.csv and DAY_AHEAD time-series files."
            )

    def get_required_columns_from_file(self, requirements_file, target_file):
        """This method reads required columns for a target input file."""
        required_files_df = pd.read_csv(requirements_file)

        row = required_files_df[required_files_df["file"] == target_file]

        if row.empty:
            raise ValueError(
                f"Could not find required column information for "
                f"{target_file} in {requirements_file}."
            )

        required_columns_str = row["required_columns"].iloc[0]

        return [col.strip() for col in required_columns_str.split(";") if col.strip()]

    def validate_required_columns(
        self,
        data_path,
        target_file,
        requirements_file=None,
    ):
        """This method validates the required columns and required row
        values before Prescient parsing.

        This method checks that each required column listed in
        required_data_files.csv exists in the target input file and
        that required columns are populated for all rows. For gen.csv,
        ramp-rate data are required only for thermal generators.

        """

        data_path = Path(data_path)
        data_file = data_path / target_file
        if requirements_file is None:
            requirements_file = data_path.parent / "required_data_files.csv"
        else:
            requirements_file = Path(requirements_file)

        if not data_file.exists():
            raise FileNotFoundError(f"Required file not found: {data_file}")

        if not requirements_file.exists():
            raise FileNotFoundError(
                f"Required data-file specification not found: {requirements_file}"
            )

        required_columns = self.get_required_columns_from_file(
            requirements_file,
            target_file,
        )

        df = pd.read_csv(data_file)

        ramp_col = "Ramp Rate MW/Min"
        thermal_unit_types = {"CT", "CC", "STEAM", "COAL", "NUC", "THERMAL"}

        def is_missing(series):
            return (
                series.isna()
                | (series.astype(str).str.strip() == "")
                | (
                    series.astype(str)
                    .str.strip()
                    .str.upper()
                    .isin(["NA", "NAN", "NONE"])
                )
            )

        # For gen.csv, Ramp Rate MW/Min is required only for thermal
        # generators.
        if target_file == "gen.csv" and ramp_col in required_columns:
            required_columns_no_ramp = [
                col for col in required_columns if col != ramp_col
            ]
        else:
            required_columns_no_ramp = required_columns

        missing_columns = [
            col for col in required_columns_no_ramp if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{target_file} is missing required column(s): "
                f"{missing_columns}. Please update {target_file} before "
                "loading data with Prescient."
            )

        # Check that each required column has populated values for all
        # rows.
        for col in required_columns_no_ramp:
            missing_mask = is_missing(df[col])

            if missing_mask.any():
                if "GEN UID" in df.columns:
                    missing_rows = df.loc[missing_mask, "GEN UID"].astype(str).tolist()
                    row_description = f"generator(s): {missing_rows}"
                elif "UID" in df.columns:
                    missing_rows = df.loc[missing_mask, "UID"].astype(str).tolist()
                    row_description = f"row UID(s): {missing_rows}"
                elif "Bus ID" in df.columns:
                    missing_rows = df.loc[missing_mask, "Bus ID"].astype(str).tolist()
                    row_description = f"bus ID(s): {missing_rows}"
                else:
                    missing_rows = df.index[missing_mask].tolist()
                    row_description = f"row index/indices: {missing_rows}"

                raise ValueError(
                    f"{target_file} has missing values in required column "
                    f"'{col}' for {row_description}. Please update the input "
                    "data before loading with Prescient."
                )

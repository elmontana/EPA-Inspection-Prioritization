drop table if exists data_exploration.aggregated_features_acs;
create table data_exploration.aggregated_features_acs as (
	select distinct f.id_number, t.*, zip_county_pop_table.county, zip_county_pop_table.county_population from data_exploration.rcra_facilities f
	left join (
		select x.zip, sum(x."B01003_001E") zip_population -- add more sums here for more fields
		from (select l.zip, a.*
		from data_exploration.acs a
		inner join data_exploration.acs_link l
		on a.state = l.state and a.county = l.county and a.tract = l.tract) x
		group by x.zip) t
	on f.zip_code = t.zip

	left join (
		select linking_table.zip, county_pop_table.* from data_exploration.acs_link linking_table
		left join(
		select a.county, sum(a."B01003_001E") county_population -- add more sums here for more fields
		from data_exploration.acs a
		group by a.county) county_pop_table
	on linking_table.county = county_pop_table.county) zip_county_pop_table
	on f.zip_code = zip_county_pop_table.zip
);



-- below is just a copy paste of above, with the first two lines changed to create in semantic schema
-- perhaps something else needs to be changed?
drop table if exists semantic.features_acs;
create table semantic.features_acs as (
	select distinct f.id_number, t.*, zip_county_pop_table.county, zip_county_pop_table.county_population from data_exploration.rcra_facilities f
	left join (
		select x.zip, sum(x."B01003_001E") zip_population -- add more sums here for more fields
		from (select l.zip, a.*
		from data_exploration.acs a
		inner join data_exploration.acs_link l
		on a.state = l.state and a.county = l.county and a.tract = l.tract) x
		group by x.zip) t
	on f.zip_code = t.zip

	left join (
		select linking_table.zip, county_pop_table.* from data_exploration.acs_link linking_table
		left join(
		select a.county, sum(a."B01003_001E") county_population -- add more sums here for more fields
		from data_exploration.acs a
		group by a.county) county_pop_table
	on linking_table.county = county_pop_table.county) zip_county_pop_table
	on f.zip_code = zip_county_pop_table.zip
);

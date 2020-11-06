create table features_acs as (
	select f.id_number, t.* from data_exploration.rcra_facilities f
	left join (
		select x.zip, sum(x."B01003_001E") zip_population -- add more sums here for more fields
		from (select l.zip, a.*
		from data_exploration.acs a
		inner join data_exploration.acs_link l
		on a.state = l.state and a.county = l.county and a.tract = l.tract) x
		group by x.zip) t
	on f.zip_code = t.zip

	left join (
		select linking_table.zip_code, county_pop_table.* from data_exploration.acs_link linking_table
		left join(
		select x.county, sum(x."B01003_001E") county_population -- add more sums here for more fields
		from data_exploration.acs a
		) county_pop_table
	on linking_table.county = county_pop_table.county) zip_county_pop_table
	on f.zip_code = zip_county_pop_table.zip_code
);
drop view if exists cleaned.{prefix}_rcra_reporting_aggregated;
create view cleaned.{prefix}_rcra_reporting_aggregated as (
    select handler_id, waste_code, sum(generation_tons) total_gen_tons from (
        select case
        when waste_code_group = 'D001' then 'd001'
        when waste_code_group = 'D002' then 'd002'
        when waste_code_group = 'D008' then 'd008'
        when waste_code_group = 'TCORICR' then 'tcor_icr'
        when waste_code_group = 'F1_5' then 'f1_5'
        when waste_code_group = 'TCMT' then 'tcmt'
        when waste_code_group = 'D011' then 'd011'
        when waste_code_group = 'D039' then 'd039'
        when waste_code_group = 'F003' then 'f003'
        when waste_code_group = 'D009' then 'd009'
        when waste_code_group = 'ICR' then 'icr'
        when waste_code_group = 'D007' then 'd007'
        when waste_code_group = 'F002' then 'f002'
        when waste_code_group = 'LABP' then 'labp'
        when waste_code_group = 'D016' then 'd016'
        when waste_code_group = 'D018' then 'd018'
        when waste_code_group = 'F005' then 'f005'
        when waste_code_group = 'F006' then 'f006'
        when waste_code_group = 'F001' then 'f001'
        when waste_code_group = 'D003' then 'd003'
        when waste_code_group = 'P001' then 'p001'
        when waste_code_group = 'F039' then 'f039'
        when waste_code_group = 'D005' then 'd005'
        when waste_code_group = 'TCOR Only' then 'tcor_only'
        when substr(waste_code_group, 1, 1) = 'D' then 'other_d'
        when substr(waste_code_group, 1, 1) = 'F' then 'other_f'
        when substr(waste_code_group, 1, 1) = 'K' then 'other_k'
        when substr(waste_code_group, 1, 1) = 'P' then 'other_p'
        when substr(waste_code_group, 1, 1) = 'U' then 'other_u'
        else 'others'
        end waste_code, * from cleaned.{prefix}_rcra_reporting) r
    group by handler_id, waste_code
    order by handler_id, waste_code
);

drop table if exists semantic.{prefix}_reporting;
create table semantic.{prefix}_reporting as (
    select *
    from crosstab(
        'select e.id as entity_id, k.waste_code, k.total_gen_tons from (
            select cr.*, case when a.total_gen_tons > 0 then a.total_gen_tons else 0 end as total_gen_tons from (
                select * from (select distinct handler_id from cleaned.{prefix}_rcra_reporting_aggregated) i cross join (
                    select distinct waste_code from cleaned.{prefix}_rcra_reporting_aggregated) c
                order by i.handler_id, c.waste_code
            ) cr
            left join cleaned.{prefix}_rcra_reporting_aggregated a
            on cr.handler_id = a.handler_id and cr.waste_code = a.waste_code) k
        inner join cleaned.{prefix}_entity_id e on k.handler_id = e.id_number'
        ) as ct (
        entity_id int,
        d001 float, d002 float, d008 float, tcor_icr float, f1_5 float,
        tcmt float, d011 float, d039 float, f003 float, d009 float,
        icr float, d007 float, f002 float, labp float, d016 float,
        d018 float, f005 float, f006 float, f001 float, d003 float,
        p001 float, f039 float, d005 float, tcor_only float, other_d float,
        other_f float, other_k float, other_p float, other_u float,
        others float
    )
);

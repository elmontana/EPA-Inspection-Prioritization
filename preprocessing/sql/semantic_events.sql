drop table if exists semantic.events;
create table semantic.events as (
    select t.id as entity_id, v.id as event_id, d.event_date, d.found_violation, d.penalty_amount from (
        select
        epa_handler_id, receive_date,
        max(greatest(evaluation_start_date, violation_determined_date, actual_return_to_compliance_date, enforcement_action_date, disposition_status_date)) as event_date,
        case when (sum(case when found_violation_flag = 'Y' then 1 else 0 end) > 0) then 1 else 0 end as found_violation,
        coalesce(max(final_amount), 0) as penalty_amount
        from cleaned.rcra_cmecomp3
        group by epa_handler_id, receive_date
        order by epa_handler_id, receive_date) as d
    inner join cleaned.entity_id t on t.id_number = d.epa_handler_id
    inner join cleaned.event_id v on v.epa_handler_id = d.epa_handler_id and v.receive_date = d.receive_date
);

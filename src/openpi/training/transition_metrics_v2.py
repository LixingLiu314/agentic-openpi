"""Causal transition scoring with explicit observation opportunities.

All labels, episode IDs and ordered stages belong to this offline scorer only.
No target labels or future boundaries are inference inputs.
"""
from collections import defaultdict

import numpy as np

VERSION = "transition_metrics_v2.0"


def causal_sample(rows, period, phase=0.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode"]].append(row)
    result = []
    for episode in sorted(groups):
        group = sorted(groups[episode], key=lambda r: r["timestamp"])
        times = np.array([r["timestamp"] for r in group])
        requests = np.arange(times[0] + phase, times[-1] + 1e-8, period)
        indices = np.searchsorted(times, requests, side="right") - 1
        result.extend(group[int(i)] for i in np.unique(indices[indices >= 0]))
    return result


def _runs(group):
    result = []
    for i, row in enumerate(group):
        if not result or row["prediction"] != group[result[-1][-1]]["prediction"]:
            result.append([i])
        else:
            result[-1].append(i)
    return result


def score_events(rows, events, hold=0.3, failure_cost=2.5):
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)
    for group in by_episode.values():
        group.sort(key=lambda r: r["timestamp"])
        if any(a["timestamp"] >= b["timestamp"] for a, b in zip(group, group[1:])):
            raise ValueError("Duplicate or unordered observation timestamps")
    details = []
    for event in events:
        group = by_episode[event["episode"]]
        opportunities = [r for r in group if event["t_label"] <= r["timestamp"] < event["new_end_time"]]
        confirmable = len(opportunities) >= 2 and opportunities[-1]["timestamp"] - opportunities[0]["timestamp"] >= hold - 1e-6
        kind = "confirmable" if confirmable else "single_or_short" if opportunities else "unobserved"
        selected = None
        transient = 0
        first_arrival = None
        for run in _runs(group):
            first = group[run[0]]
            if first["prediction"] != event["new"]:
                continue
            if not event["cell_start"] <= first["timestamp"] < event["cell_end"]:
                continue
            if first_arrival is None:
                first_arrival = first["timestamp"]
            confirmations = [group[i]["timestamp"] for i in run[1:]
                             if group[i]["timestamp"] - first["timestamp"] >= hold - 1e-6
                             and group[i]["timestamp"] < event["new_end_time"]]
            if confirmations and selected is None:
                selected = (first["timestamp"], confirmations[0])
            elif not confirmations:
                transient += 1
        single_correct = bool(opportunities and all(r["prediction"] == event["new"] for r in opportunities))
        status = ("confirmed" if selected else "model_missed_or_unstable") if confirmable else (
            "observed_correct_unconfirmed" if single_correct else "observed_wrong_unconfirmable"
        ) if opportunities else "sampling_unobserved"
        error = selected[0] - event["t_label"] if selected else None
        details.append({"event_id":event["event_id"], "episode":event["episode"], "task":event["task"],
                        "transition":event["transition"], "opportunity":kind, "observations":len(opportunities),
                        "status":status, "first_arrival":first_arrival, "stable_arrival":selected[0] if selected else None,
                        "confirmation":selected[1] if selected else None, "signed_label_error":error,
                        "transient_correct_runs":transient, "physical_readiness_evaluated":False})
    def summarize(items):
        eligible = [r for r in items if r["opportunity"] == "confirmable"]
        success = [r for r in eligible if r["status"] == "confirmed"]
        errors = [r["signed_label_error"] for r in success]
        costs = [min(failure_cost, max(0.0, r["signed_label_error"])) if r["status"] == "confirmed" else failure_cost for r in eligible]
        short = [r for r in items if r["opportunity"] == "single_or_short"]
        return {"events":len(items), "confirmable":len(eligible), "confirmed":len(success),
                "model_missed_or_unstable":len(eligible)-len(success),
                "confirmed_rate":len(success)/len(eligible) if eligible else None,
                "short_observed":len(short), "short_correct":sum(r["status"] == "observed_correct_unconfirmed" for r in short),
                "sampling_unobserved":sum(r["opportunity"] == "unobserved" for r in items),
                "late_p50_p95":np.percentile([max(0.0,e) for e in errors],[50,95]).tolist() if errors else None,
                "early_over_100ms":sum(e < -0.100001 for e in errors),
                "capped_late_cost_mean":float(np.mean(costs)) if costs else None,
                "failure_cost_seconds":failure_cost,
                "transient_correct_runs":sum(r["transient_correct_runs"] for r in items)}
    return {**summarize(details), "events_detail":details,
            "by_task_transition":{key:summarize([r for r in details if (r["task"],r["transition"]) == key])
                                  for key in sorted(set((r["task"],r["transition"]) for r in details))}}


def frame_and_order_metrics(rows):
    by_episode = defaultdict(list)
    for r in rows:
        by_episode[r["episode"]].append(r)
    episodes = []
    for episode, group in sorted(by_episode.items()):
        group.sort(key=lambda r:r["timestamp"])
        stages = list(dict.fromkeys(r["label"] for r in group))
        order = {s:i for i,s in enumerate(stages)}
        changes = skips = reversals = invalid = far = 0
        for i,r in enumerate(group):
            far += r["prediction"] not in order or abs(order.get(r["prediction"],-100)-order[r["label"]]) >= 2
            if i and r["prediction"] != group[i-1]["prediction"]:
                changes += 1
                a,b = order.get(group[i-1]["prediction"]),order.get(r["prediction"])
                if a is None or b is None:
                    invalid += 1
                else:
                    skips += b > a+1
                    reversals += b < a
        def counts(condition):
            selected=[r for r in group if condition(r)]
            return {"correct":sum(r["prediction"] == r["label"] for r in selected),"total":len(selected)}
        episodes.append({"episode":episode,"task":group[0]["task"],"all":counts(lambda _:True),
                         "boundary":counts(lambda r:r["boundary_event"] is not None),
                         "stable":counts(lambda r:r["boundary_event"] is None),
                         "changes":changes,"forward_jumps":skips,"backward_changes":reversals,
                         "invalid_changes":invalid,"far_stage_frames":int(far)})
    result={"episodes":episodes}
    for name in ["all","boundary","stable"]:
        denominator=sum(e[name]["total"] for e in episodes)
        result[name+"_em"]=sum(e[name]["correct"] for e in episodes)/max(1,denominator)
        result[name+"_episode_mean_em"]=float(np.mean([e[name]["correct"]/e[name]["total"] for e in episodes if e[name]["total"]]))
    for name in ["changes","forward_jumps","backward_changes","invalid_changes","far_stage_frames"]:
        result[name]=sum(e[name] for e in episodes)
    return result


def report(rows, events):
    """Caller supplies independently generated causal rollout per deployment rate.

    Never subsample a dense stateful model rollout to represent a low-rate run.
    """
    event_report=score_events(rows,events)
    event_report["by_task_transition"]={" | ".join(k):v for k,v in event_report["by_task_transition"].items()}
    return {"version":VERSION,"scope":"annotation boundary capability; physical readiness unreviewed",
            "frames":frame_and_order_metrics(rows),"events":event_report}


def paired_boundary_bootstrap(candidate, baseline, draws=10000):
    c={r["episode"]:r for r in candidate["frames"]["episodes"]}
    b={r["episode"]:r for r in baseline["frames"]["episodes"]}
    if c.keys()!=b.keys():
        raise ValueError("Paired episodes differ")
    deltas=[]
    for ep in sorted(c):
        x,y=c[ep]["boundary"],b[ep]["boundary"]
        if x["total"]!=y["total"] or not x["total"]:
            raise ValueError("Paired boundary frames differ")
        deltas.append((x["correct"]-y["correct"])/x["total"])
    delta=np.array(deltas)
    samples=delta[np.random.default_rng(42).integers(len(delta),size=(draws,len(delta)))].mean(1)
    return {"unit":"paired episode", "episodes":len(delta), "seed":42,"draws":draws,
            "mean_boundary_em_gain":float(delta.mean()),"ci95":np.percentile(samples,[2.5,97.5]).tolist()}


def semantic_gates(candidate, baseline):
    """Preregistered gates; test all rates, never improve latency by dropping failures."""
    paired=paired_boundary_bootstrap(candidate["dense"],baseline["dense"])
    c,b=candidate["dense"]["frames"],baseline["dense"]["frames"]
    gates={"boundary_gain_8pp":c["boundary_em"] >= b["boundary_em"]+.08,
           "paired_boundary_ci_positive":paired["ci95"][0]>0,
           "stable_em_guard":c["stable_em"]>=b["stable_em"]-.01,
           "far_stage_guard":c["far_stage_frames"]<=b["far_stage_frames"]}
    for key in baseline:
        if key not in candidate:raise ValueError("Missing evaluation rate "+key)
        c,b=candidate[key]["events"],baseline[key]["events"]
        if c["confirmable"]!=b["confirmable"] or c["events"]!=b["events"]:
            raise ValueError("Evaluation opportunities differ")
        gates[key+"_failure_guard"]=c["model_missed_or_unstable"]<=b["model_missed_or_unstable"]
        gates[key+"_short_guard"]=c["short_correct"]>=b["short_correct"]
        gates[key+"_early_guard"]=c["early_over_100ms"]<=b["early_over_100ms"]+.02*b["confirmable"]
        for metric in ["forward_jumps","backward_changes","invalid_changes"]:
            gates[key+"_"+metric]=candidate[key]["frames"][metric]<=baseline[key]["frames"][metric]
    low=[k for k in baseline if k.startswith("low_")]
    gates["low_rate_failure_inclusive_delay_reduction_30percent"]=bool(low) and bool(np.mean([
        candidate[k]["events"]["capped_late_cost_mean"] for k in low])<=.7*np.mean([
        baseline[k]["events"]["capped_late_cost_mean"] for k in low]))
    return {"passed":all(gates.values()),"checks":gates,"paired_bootstrap":paired,
            "action_gate_required":True,"physical_success_claimed":False}

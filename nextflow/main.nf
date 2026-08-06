#!/usr/bin/env nextflow
/*
 * XRSpinoPelvic1K -- unified spinopelvic landmark pipeline.
 *
 *   generate  ->  DRRs + projected landmarks, one task per CT (parallel)
 *   train     ->  ONE model, two sources, disjoint channels:
 *                   hip point  from DRRs (3-D sphere fit, projected)
 *                   corners    from BUU  (radiologist annotations)
 *   measure   ->  run on real laterals, QC-gate every case, report PI
 *   validate  ->  classical circle fit as an INDEPENDENT hip reference
 *
 * Nextflow rather than a chain of sbatch calls because generation is per-CT
 * embarrassingly parallel and its failures are per-case: -resume re-runs only the
 * cases that failed, instead of the whole stage. Every process is idempotent, so a
 * resumed run is the same command.
 *
 *   nextflow run nextflow/main.nf -profile slurm --ct_dir /data/ct \
 *       --label_dir /data/labels --buu /data/BUU-LSPINE_400 --outdir /data/xrsp1k
 */

nextflow.enable.dsl = 2

params.ct_dir    = null
params.label_dir = null
params.buu       = null
params.outdir    = "results"
params.n_views   = 8
params.spacing   = 1.0
params.epochs    = 150
params.ostk      = "/opt/ostk"

process GENERATE {
    tag   "${case_id}"
    label 'cpu'
    publishDir "${params.outdir}/views", mode: 'copy'

    input:
    tuple val(case_id), path(ct), path(label)

    output:
    tuple val(case_id), path("${case_id}"), emit: views

    script:
    """
    python /workspace/scripts/generate_case.py \
        --ct ${ct} --label ${label} --out ${case_id} \
        --n_views ${params.n_views} --spacing ${params.spacing} \
        --ostk_path ${params.ostk}
    """
}

process TRAIN_UNIFIED {
    tag   "unified"
    label 'gpu'
    publishDir "${params.outdir}/runs", mode: 'copy'

    input:
    path views_dir
    path buu

    output:
    path "unified", emit: run

    script:
    """
    python /workspace/scripts/train_unified.py \
        --drr ${views_dir} --buu ${buu} --out unified \
        --epochs ${params.epochs}
    """
}

process MEASURE {
    tag   "measure"
    label 'gpu'
    publishDir "${params.outdir}/results", mode: 'copy'

    input:
    path run
    path buu

    output:
    path "buu_pi.csv"

    script:
    """
    python /workspace/scripts/measure_pi_unified.py \
        --buu ${buu} --model ${run}/best.pt --out buu_pi.csv
    """
}

process VALIDATE_HIP {
    tag   "hipfit"
    label 'cpu'
    publishDir "${params.outdir}/results", mode: 'copy'

    input:
    path run
    path buu

    output:
    path "hip_circlefit.csv"

    script:
    """
    python /workspace/scripts/validate_hip_circlefit.py \
        --buu ${buu} --model ${run}/best.pt --out hip_circlefit.csv
    """
}

workflow {
    if( !params.ct_dir || !params.label_dir || !params.buu )
        error "need --ct_dir, --label_dir and --buu"

    cts = Channel.fromPath("${params.ct_dir}/*_ct.nii.gz")
        .map { f -> tuple(f.name.replaceAll('_ct\.nii\.gz$',''), f) }
    labs = Channel.fromPath("${params.label_dir}/*_label.nii.gz")
        .map { f -> tuple(f.name.replaceAll('_label\.nii\.gz$',''), f) }
    pairs = cts.join(labs)

    GENERATE(pairs)
    views = GENERATE.out.views.map { id, d -> d }.collect()
    buu   = Channel.fromPath(params.buu)

    TRAIN_UNIFIED(views, buu)
    MEASURE(TRAIN_UNIFIED.out.run, buu)
    VALIDATE_HIP(TRAIN_UNIFIED.out.run, buu)
}

# SAE Training Log

This log summarises the main engineering changes I added while turning an initial smoke test into a usable SAE training pipeline. I group related changes together, while keeping the overall development order.

## 1. Establishing a runnable training path

I first focused on getting a minimal end-to-end path to run on Katana: environment setup, model access, cache handling, and a smoke test that could load the base model, hook one layer, extract activations, and run a minimal SAE training step. At this stage, the goal was not performance; it was to confirm that the whole path from text to activations to SAE updates was alive.

I then split the project into separate modules for configuration, distributed utilities, data loading, activation extraction, initial statistics, the SAE model, and the main training loop. This made later debugging and incremental changes much safer than continuing from a single expanding script.

## 2. Turning the smoke test into a structured SAE pipeline

After the minimal run succeeded, I formalised the training structure. I froze the base language model and used it only as an activation generator. I added a dedicated activation capture component, filtered unusable special-token positions, and built a proper training loop around the SAE rather than treating the whole system as a one-off experiment.

I also added checkpoint saving and periodic logging so that long runs on HPC could be monitored and resumed from meaningful intermediate states. This became important once training moved beyond short tests.

## 3. Improving initialisation and startup stability

A major early refinement was decoder-bias initialisation. Instead of starting `b_dec` from zero, I changed the pipeline to initialise it from the mean activation vector estimated from the hooked model activations. This was one of the first changes intended to improve training stability rather than just make the code run.

That initialisation step then evolved further. The mean activation and related initial statistics were moved into a dedicated pre-training stage, and later cached so they did not need to be recomputed every time training started. This was especially useful on Katana, where startup time became a noticeable bottleneck.

I also added basic token-length profiling and activation-scale estimation so that sequence-length choices and activation normalisation were informed by observed data rather than guesswork.

## 4. Making the data path more coherent

As the project grew, I found that the codebase was drifting between multiple data-loading ideas: direct streaming, local cache expansion, and partially overlapping pathways. This was creating avoidable confusion and version drift.

I therefore consolidated the project around a single main data path based on streamed text batches and downstream activation extraction. This was an important engineering cleanup step because it reduced technical debt and made subsequent optimisation work much easier.

## 5. Feeding the SAE more effectively

Once the core path was stable, I added an activation buffer so that SAE updates were performed on buffered activations rather than on whatever small fragment happened to be produced by the latest model forward pass. This separated activation collection from SAE optimisation more cleanly and made the training loop more structured.

I also added per-rank text batching for distributed runs, so that multi-GPU training did not simply duplicate the same text processing on every rank.

## 6. Scheduler and training-control refinements

After the basic pipeline was working, I added learning-rate warmup followed by cosine decay, and I also added warmup for the sparsity coefficient. These were introduced after the project had already reached the stage where training could proceed for meaningful numbers of steps; the purpose was to make training dynamics less abrupt and more controlled during longer runs.

I also added decoder normalisation-related maintenance steps during training, so that the SAE parameters were kept in a more controlled regime across updates.

## 7. CPU/GPU utilisation: problem, intervention, feedback

A major engineering focus later in the project was poor utilisation. In practice, I observed that GPU usage was low, CPU usage was also low, and the input side of the training loop appeared under-active. This suggested that the system was not compute-bound in the intended way; instead, the training loop was frequently waiting on upstream work.

I treated this as an iterative optimisation problem rather than a single bug.

First, I identified that startup latency and data delivery were likely contributors. The initial-statistics stage was taking too long, so I changed it from always recomputing to a cached load-if-available workflow. That reduced repeated startup cost.

Second, I focused on the text-input side. I added configurable prefetching for text batches, with thread/process-based options, so that data preparation could overlap better with downstream training. This was a direct response to the low-utilisation symptoms.

Third, I reviewed the broader data path and removed mixed loading patterns where possible. The rationale was that utilisation problems are often amplified by architectural inconsistency: when there is no single clean path for data flow, it becomes harder to reason about where the actual bottleneck sits.

The project therefore moved through a repeated loop:

- observe low GPU and CPU utilisation,
- identify likely pipeline stalls,
- simplify or precompute expensive stages,
- add prefetching and buffering,
- then reassess behaviour.

This part of the work mattered because the problem was not merely speed in the abstract; it was whether the training system was using the available hardware in a way that justified longer Katana runs.

## 8. Current state of the training pipeline

At the current stage, the project has moved well beyond a smoke test. It now has a modular training structure, frozen-model activation extraction, a unified streamed data path, initial-statistics caching, mean-based decoder-bias initialisation, activation buffering, distributed training support, training schedulers, checkpointing, and periodic logging.

In short, the project evolved from “can this run at all?” to “can this run in a repeatable, monitorable, and increasingly efficient way on HPC?”

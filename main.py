import argparse
import sys
import logging
import shutil
from pathlib import Path
from core.api.compiler_api import NovelFactoryAPI

# Force unbuffered output so every print/log line appears immediately in Kaggle
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("NovelFactory")

def handle_compile(api: NovelFactoryAPI, args: argparse.Namespace):
    if args.model:
        api.use_model(args.model)
    # Could handle args.preset, args.seed, etc. here
    api.compile(target=args.target, resume=args.resume)

def handle_status(api: NovelFactoryAPI, args: argparse.Namespace):
    status_data = api.status()
    print("="*50)
    print("PROJECT STATUS")
    print("="*50)
    for k, v in status_data.items():
        if isinstance(v, dict):
            print(f"{k.capitalize()}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k.capitalize()}: {sub_v}")
        else:
            print(f"{k.capitalize().replace('_', ' ')}: {v}")

def handle_doctor(api: NovelFactoryAPI, args: argparse.Namespace):
    report = api.doctor().model_dump()
    print("="*50)
    print("NOVELFACTORY DOCTOR")
    print("="*50)
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")

def handle_explain(api: NovelFactoryAPI, args: argparse.Namespace):
    report = api.explain(args.target)
    print("="*50)
    print(f"PROVENANCE: {args.target}")
    print("="*50)
    for step in report["trace"]:
        print(f"↓ {step}")

def handle_benchmark(api: NovelFactoryAPI, args: argparse.Namespace):
    report = api.benchmark().model_dump()
    import json
    print(json.dumps(report, indent=2))

def handle_logs(api: NovelFactoryAPI, args: argparse.Namespace):
    print(f"Fetching logs for {args.type}...")
    # Read from reports/

def handle_project(api: NovelFactoryAPI, args: argparse.Namespace):
    api.project_action(args.action)

def handle_workspace(api: NovelFactoryAPI, args: argparse.Namespace):
    api.workspace_action(args.action)

def handle_inspect(api: NovelFactoryAPI, args: argparse.Namespace):
    print(api.inspect(args.target))

def handle_graph(api: NovelFactoryAPI, args: argparse.Namespace):
    api.graph(args.view)

def handle_cache(api: NovelFactoryAPI, args: argparse.Namespace):
    api.cache_action(args.action)

def handle_assets(api: NovelFactoryAPI, args: argparse.Namespace):
    api.assets_action(args.action)

def handle_models(api: NovelFactoryAPI, args: argparse.Namespace):
    api.models_action(args.action)

def handle_export(api: NovelFactoryAPI, args: argparse.Namespace):
    api.export(args.format)


# ─────────────────────────────────────────────────────────────────────────────
# --novel shorthand handler
# ─────────────────────────────────────────────────────────────────────────────
def handle_novel_shorthand(novel_path: str, stage: str, resume: bool, model: str | None):
    """
    Convenience entry-point for:
        python main.py --novel my_script.txt --stage all

    What it does:
      1. Resolves the novel file (absolute or relative to cwd).
      2. Creates a project directory next to the file named after its stem
         (e.g.  my_script.txt  →  ./my_script_project/).
      3. Copies the .txt into that project dir if it isn't already there.
      4. Runs compile(target=stage) exactly like the subcommand would.
    """
    src = Path(novel_path).resolve()
    if not src.exists():
        logger.error(f"Novel file not found: {src}")
        sys.exit(1)

    # Project dir lives alongside the novel file
    project_dir = src.parent / f"{src.stem}_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Copy novel into project dir so _load_novel_text() finds it
    dest = project_dir / src.name
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest)
        logger.info(f"Copied novel → {dest}")

    logger.info(f"Project dir : {project_dir}")
    logger.info(f"Stage       : {stage}")

    api = NovelFactoryAPI(project_dir=str(project_dir))
    if model:
        api.use_model(model)
    api.compile(target=stage, resume=resume)


def main():
    parser = argparse.ArgumentParser(
        description="NovelFactory Digital Film Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start (one-liner):
  python main.py --novel my_script.txt --stage all

Full subcommand interface:
  python main.py --project ./my_project compile all
  python main.py --project ./my_project status
  python main.py --project ./my_project doctor
        """,
    )

    # ── Global flags ──────────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=None,
                        help="Path to project directory (contains the .txt novel file)")

    # ── --novel / --stage shorthand ───────────────────────────────────────────
    parser.add_argument("--novel", type=str, default=None,
                        metavar="FILE",
                        help="Path to novel .txt file. Auto-creates a project dir next to it.")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["plan", "render", "assemble", "all"],
                        help="Pipeline stage to run (default: all). Used with --novel.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last interrupted state")
    parser.add_argument("--model", type=str, default=None,
                        help="Diffusion model override (e.g. sdxl-lightning)")

    subparsers = parser.add_subparsers(dest="command")

    # compile
    compile_parser = subparsers.add_parser("compile", help="Compile the project")
    compile_parser.add_argument("target", choices=["plan", "render", "assemble", "all"],
                                help="Target stage to compile")
    compile_parser.add_argument("--resume", action="store_true",
                                help="Resume from last interrupted state")
    compile_parser.add_argument("--model", type=str,
                                help="Provider model to use (e.g. sdxl-lightning)")
    compile_parser.add_argument("--preset", type=str, help="Render preset to use")
    compile_parser.add_argument("--seed", type=int, help="Global random seed")
    compile_parser.add_argument("--jobs", type=int, help="Number of parallel jobs")
    compile_parser.set_defaults(func=handle_compile)

    # status
    status_parser = subparsers.add_parser("status", help="Show project dashboard")
    status_parser.set_defaults(func=handle_status)

    # doctor
    doctor_parser = subparsers.add_parser("doctor", help="Run Kaggle environment diagnostics")
    doctor_parser.set_defaults(func=handle_doctor)

    # explain
    explain_parser = subparsers.add_parser("explain", help="Show provenance of an asset/manifest")
    explain_parser.add_argument("target", type=str, help="ID of the asset or manifest")
    explain_parser.set_defaults(func=handle_explain)

    # benchmark
    benchmark_parser = subparsers.add_parser("benchmark",
                                             help="Show comprehensive compiler performance metrics")
    benchmark_parser.set_defaults(func=handle_benchmark)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Access reports and logs")
    logs_parser.add_argument("type", choices=["latest", "benchmark", "doctor", "validation"],
                             help="Log type to view")
    logs_parser.set_defaults(func=handle_logs)

    # project
    project_parser = subparsers.add_parser("project", help="Project management")
    project_parser.add_argument("action",
                                choices=["init", "open", "info", "clean", "archive", "clone", "list"],
                                help="Action")
    project_parser.set_defaults(func=handle_project)

    # workspace
    workspace_parser = subparsers.add_parser("workspace",
                                             help="Workspace health and folder operations")
    workspace_parser.add_argument("action", choices=["info", "verify", "clean", "repair"],
                                  help="Action")
    workspace_parser.set_defaults(func=handle_workspace)

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Universal object inspector")
    inspect_parser.add_argument("target", type=str, help="Target ID (e.g., shot_018, asset:123)")
    inspect_parser.set_defaults(func=handle_inspect)

    # graph
    graph_parser = subparsers.add_parser("graph", help="Generate DOT graph visualizations")
    graph_parser.add_argument("view",
                              choices=["pipeline", "scene", "render", "assets", "state", "dependencies"],
                              help="Graph view")
    graph_parser.set_defaults(func=handle_graph)

    # cache
    cache_parser = subparsers.add_parser("cache", help="Manage caches")
    cache_parser.add_argument("action", choices=["stats", "verify", "clean", "repair"],
                              help="Action")
    cache_parser.set_defaults(func=handle_cache)

    # assets
    assets_parser = subparsers.add_parser("assets", help="Manage CAS assets")
    assets_parser.add_argument("action", choices=["list", "verify", "orphaned", "export", "lineage"],
                               help="Action")
    assets_parser.set_defaults(func=handle_assets)

    # models
    models_parser = subparsers.add_parser("models", help="Manage provider models")
    models_parser.add_argument("action",
                               choices=["list", "info", "download", "verify", "cache", "remove", "benchmark"],
                               help="Action")
    models_parser.set_defaults(func=handle_models)

    # export
    export_parser = subparsers.add_parser("export", help="Export artifacts")
    export_parser.add_argument("format",
                               choices=["video", "images", "project", "report", "manifests"],
                               help="Format to export")
    export_parser.set_defaults(func=handle_export)

    args = parser.parse_args()

    # ── Route: --novel shorthand takes priority over subcommands ─────────────
    if args.novel:
        handle_novel_shorthand(
            novel_path=args.novel,
            stage=args.stage,
            resume=args.resume,
            model=args.model,
        )
        return

    # ── Route: subcommand interface ───────────────────────────────────────────
    if args.command is None:
        parser.print_help()
        return

    project_dir = args.project or "workspace"
    api = NovelFactoryAPI(project_dir=project_dir)
    if hasattr(args, "func"):
        args.func(api, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()



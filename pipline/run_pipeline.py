"""
run_pipeline.py — Master runner for Paper 2 data collection
Usage:
    python run_pipeline.py prices          # fetch commodity prices
    python run_pipeline.py news            # fetch news from all sources
    python run_pipeline.py classify        # run LLM classification (needs ANTHROPIC_API_KEY)
    python run_pipeline.py classify --dry-run
    python run_pipeline.py all             # full pipeline
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Paper 2 Data Pipeline")
    parser.add_argument("stage", choices=["prices", "news", "classify", "event_study", "egarch", "all"])
    parser.add_argument("--freq", default="1d", help="Price frequency: 1d, 1h")
    parser.add_argument("--max-pages", type=int, default=10, help="News pages per query")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0, help="Event study sigma threshold")
    args = parser.parse_args()

    if args.stage in ("prices", "all"):
        print("\n" + "▓" * 60)
        print("  STAGE: Fetch Prices")
        print("▓" * 60)
        from src.data_collection.fetch_prices import fetch_all
        fetch_all(interval=args.freq)

    if args.stage in ("news", "all"):
        print("\n" + "▓" * 60)
        print("  STAGE: Fetch Yahoo News")
        print("▓" * 60)
        from src.data_collection.fetch_news_yahoo import fetch_all_news
        fetch_all_news(max_pages=args.max_pages)

        print("\n" + "▓" * 60)
        print("  STAGE: Fetch ForexFactory Calendar")
        print("▓" * 60)
        from src.data_collection.fetch_news_forexfactory import fetch_calendar
        fetch_calendar()

    if args.stage in ("classify", "all"):
        import os
        input_path = os.path.join("data", "raw", "news", "yahoo_news_raw.json")
        if not os.path.exists(input_path):
            print(f"ERROR: {input_path} not found. Run 'news' stage first.")
            sys.exit(1)
        print("\n" + "▓" * 60)
        print("  STAGE: LLM Classification → TPSI")
        print("▓" * 60)
        from src.llm_pipeline.llm_classify import run_pipeline
        run_pipeline(input_path, dry_run=args.dry_run)

    if args.stage in ("event_study", "all"):
        print("\n" + "▓" * 60)
        print("  STAGE: Event Study (CAR)")
        print("▓" * 60)
        from src.analysis.event_study import run_event_study
        run_event_study(threshold_sigma=args.threshold)


    if args.stage in ("egarch", "all"):
        print("\n" + "▓" * 60)
        print("  STAGE: EGARCH(1,1) Analysis")
        print("▓" * 60)
        from src.analysis.egarch import run_egarch
        run_egarch()


if __name__ == "__main__":
    main()

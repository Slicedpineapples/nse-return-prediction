# NSE CLI — bash completion
#
# Installation (pick one):
#
#   Temporary (current shell):
#     source /path/to/nse_completions.bash
#
#   Permanent (add to ~/.bashrc or ~/.bash_profile):
#     source /path/to/ml.nse/scripts/completions/nse_completions.bash
#
#   System-wide (macOS Homebrew bash-completion@2):
#     cp nse_completions.bash "$(brew --prefix)/etc/bash_completion.d/nse"

_nse_date_hint() {
    COMPREPLY=( "$(date +%Y-%m-%d)" )
}

_nse_complete() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    case "${COMP_WORDS[0]}" in

        nse-ingest)
            case "$prev" in
                --start|--end) _nse_date_hint; return ;;
            esac
            COMPREPLY=( $(compgen -W "--start --end --force --verbose -v --help" -- "$cur") )
            ;;

        nse-features)
            case "$prev" in
                --start|--end) _nse_date_hint; return ;;
            esac
            COMPREPLY=( $(compgen -W "--start --end --force --verbose -v --help" -- "$cur") )
            ;;

        nse-train)
            COMPREPLY=( $(compgen -W "--n-months --prod-only --recency-half-life --verbose -v --help" -- "$cur") )
            ;;

        nse-backtest)
            case "$prev" in
                --start|--end) _nse_date_hint; return ;;
                --scenario) COMPREPLY=( $(compgen -W "optimistic base stressed" -- "$cur") ); return ;;
            esac
            COMPREPLY=( $(compgen -W "--start --end --model --scenario --verbose -v --help" -- "$cur") )
            ;;

        nse-signal)
            case "$prev" in
                --date) _nse_date_hint; return ;;
            esac
            COMPREPLY=( $(compgen -W "--date --model --skip-ingest --skip-llm --verbose -v --help" -- "$cur") )
            ;;

        nse-pipeline)
            case "$prev" in
                --date) _nse_date_hint; return ;;
            esac
            COMPREPLY=( $(compgen -W "--date --model --skip-ingest --skip-llm --verbose -v --help" -- "$cur") )
            ;;

        nse-monitor)
            case "$prev" in
                --start|--end) _nse_date_hint; return ;;
            esac
            COMPREPLY=( $(compgen -W "--start --end --window --backfill --drift --verbose -v --help" -- "$cur") )
            ;;

    esac
}

complete -F _nse_complete nse-ingest
complete -F _nse_complete nse-features
complete -F _nse_complete nse-train
complete -F _nse_complete nse-backtest
complete -F _nse_complete nse-signal
complete -F _nse_complete nse-pipeline
complete -F _nse_complete nse-monitor

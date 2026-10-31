# Sourced by the test scripts (bash or zsh) with the tests directory as its argument: exports the variables of tests/local.env and tests/models.env
# that the environment does not set (environment > local.env > models.env, as test_paths.h reads them) and defines `model_dir ROLE`, the
# absolute checkpoint directory of a role of models.env (its value is a directory under ENGINE_MODEL_ROOT or an absolute path).
_models_env_load() {
	local line k v cur
	while IFS= read -r line || [ -n "$line" ]; do
		line=${line%%#*}
		line=${line#export }
		case $line in
			*=*)
				k=${line%%=*}; v=${line#*=}
				case $k in ''|*[!A-Za-z0-9_]*) continue ;; esac
				v=${v%\"}; v=${v#\"}; v=${v%\'}; v=${v#\'}
				eval "cur=\${$k-}"
				[ -z "$cur" ] && export "$k=$v"
				;;
		esac
	done < "$1"
	return 0
}
[ -f "$1/local.env" ] && _models_env_load "$1/local.env"
_models_env_load "$1/models.env"
model_dir() {
	local v
	eval "v=\${$1-}"
	case $v in
		/*) printf '%s\n' "$v" ;;
		*) printf '%s\n' "$ENGINE_MODEL_ROOT/$v" ;;
	esac
}

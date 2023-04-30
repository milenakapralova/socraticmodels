import ruamel.yaml

def remove_conda_forge_deps(yml_file):
    with open(yml_file) as f:
        env = ruamel.yaml.safe_load(f)

    deps = env.get('dependencies', [])

    for i in range(len(deps)):
        if isinstance(deps[i], str):
            deps[i] = '='.join(deps[i].split('=')[:2])

    env['dependencies'] = [dep for dep in deps if not isinstance(dep, dict)]

    with open(yml_file, 'w') as f:
        ruamel.yaml.dump(env, f, Dumper=ruamel.yaml.RoundTripDumper)

if __name__ == '__main__':
    remove_conda_forge_deps('environment2.yml')

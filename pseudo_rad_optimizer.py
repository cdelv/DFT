#########################################################################
# @Copyright Carlos Andrés del Valle. cdelv@unal.edu.co. Jul 4, 2024.
#########################################################################
import glob
import os
import shutil
import argparse
import json
import numpy as np
import pandas as pd
from scipy import optimize

Configuration = {
    'input_psuedo' : '',
    'input_test' : '',
    'pg' : '/opt/Atom/atom-4.2.7-100/Tutorial/Utils/pg.sh',
    'pt' : '/opt/Atom/atom-4.2.7-100/Tutorial/Utils/pt.sh',
    'script_dir' : os.path.abspath(os.getcwd()),
    'out_dir' : 'Out',
    'data_dir' : 'Data',
    'data_pfile' : '',
    'data_tfile' : '',
    'opvars' : [],
    'maxiter' : 1500,
    'bounds' : [
        [0.0, 2.5], # s
        [0.0, 3.0], # p
        [0.0, 3.5], # d
        [0.0, 4.0], # f
        [0.0, 3.0]  # rcore
    ],
    'a' : 0.45,
    'b' : 1.0,
    'q0' : 25.2,
    'pseudo_config' : {
        'run_type': 'pg',
        'title': 'Sample H pseudo',
        'PS_flavor': 'tm2',
        'logder_r': 2.0,
        'chemical_symbol': 'H',
        'XC_flavor': 'ca',
        'XC_id' : '00000000',
        'spin_relativistic': '',
        'fractional_atomic_number': 0.0,
        'norbs_core': 0,
        'norbs_valence': 1,
        'rs': 0.1,
        'rp': 0.1,
        'rd': 0.1,
        'rf': 0.1,
        'r_core_flag': 1.0,
        'rcore': 0.0,
        'orbital_block': [[1, 0, 0.5, 0]]
    },
    'tests' : []
}
Orbital_Keys = {0 : 'rs', 1 : 'rp', 2 : 'rd', 3 : 'rf', 4 : 'rcore'}
Orbital_Keys_Rev =  {'rs' : 0, 'rp' : 1, 'rd' : 2, 'rf' : 3, 'rcore' : 4}
Bad_Return = 10 + 10 * Configuration['b']
Count = 0

def hardness_penalization(qmax):
    return (Configuration['b'] /(1.0 + np.exp((Configuration['q0'] - qmax)/Configuration['a'])))

def main():
    parser = argparse.ArgumentParser(
        prog = '',
        description = '''
        This program uses the annealing optimization algorithm to find the best combination of cutoff radii to create the pseudopotential.
        It uses the transferability test to evaluate how "good" a pseudopotential is.
        To use it, you need to provide the path of the Atom pg.sh and pt.sh scripts and have a pseudo input file and a transferability test file.
        Read the rest of the flags for details. The program also creates a config.json file to load an old state (overrides all passed varaibles). This is used for -res and -op with --resume.
        ''',
        epilog =
        '''
        Although the program sanitizes your input files, you still requires the input to be in the format expected by the Atom program.
        Report bugs to cdelv@unal.edu.co.
        '''
    )
    functions_group = parser.add_mutually_exclusive_group(required=True)
    functions_group.add_argument('-cf', '--check-files', action='store_true', help='Check that all necessary files exist.')
    functions_group.add_argument('-op', '--optimize', action='store_true', help='Optimize the cutoff radii of the given pseudopotential using the given transferability test.')
    functions_group.add_argument('-res', '--results', action='store_true', help='Recover results from the out.log file in --out_dir and generate inside --out_dir the corresponding pseudo files and plots.')
    functions_group.add_argument('-eval', '--evaluate', action='store_true', help='Evaluate the pseudopotential with given cutoff radii. Not given cutoff radii default to the values present in the input pseudo file.')

    conf_group = parser.add_argument_group('Program configuration variables')
    conf_group.add_argument('-ip', '--input_psuedo', type=str, help='Input pseudo file to use. If not set, it looks for a *.inp file in the execution directory.')
    conf_group.add_argument('-it', '--input_test', type=str, help='Input test file to use. If not set, it looks for a *test* file in the execution directory.')
    conf_group.add_argument('-pg', type=str, default=Configuration['pg'], help='Path to pg.sh Atom script. Defaults to '+Configuration['pg'])
    conf_group.add_argument('-pt', type=str, default=Configuration['pt'], help='Path to pt.sh Atom script. Defaults to '+Configuration['pt'])
    conf_group.add_argument('-o', '--out_dir', type=str, default=Configuration['out_dir'], help='Name of the output directory. Here the optimization log and results will be stored. Defaults to '+Configuration['out_dir'])
    conf_group.add_argument('-d', '--data_dir', type=str, default=Configuration['data_dir'], help='Name of the data directory. Here the pseudos are generated and hold temporary files. It gets ovewritten each time a new pseudo is created. Defaults to '+Configuration['data_dir'])
    conf_group.add_argument('-rcf', '--r_core_flag', type=float, default=Configuration['pseudo_config']['r_core_flag'], help='Value of the r_core_flag to use. The program does not optimize this value. Defaults to '+str(Configuration['pseudo_config']['r_core_flag']))

    variables_group = parser.add_argument_group('Program evaluation variables')
    variables_group.add_argument('-rs', type=float, help='Value of the s orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rp', type=float, help='Value of the p orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rd', type=float, help='Value of the d orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rf', type=float, help='Value of the f orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rc', '--rcore', type=float, help='Value of the rcore cutoff. If set in an evaluation run, overrides the value present in the input pseudo *.inp file. Only used in corre correction runs (pe instead of pg)')

    op_variables_group = parser.add_argument_group('Program optimization variables')
    op_variables_group.add_argument('--resume', action='store_false', help='Perform an optimization run without deleting previous results.')
    op_variables_group.add_argument('-opvar', '--optimization_variables', nargs='*', type=str, choices=['rs', 'rp', 'rd', 'rf', 'rcore'], help="Optimize the pseudopotential using only the selected orbitals' cutoff radius. Defaults to the orbitals detected in the pseudo. For the fixed orbitals, it uses the value in the input *.inp pseudo file or the value set by the -r* flags. If core corrections, the progrma includes rc in the default orbitals to optimize.")
    op_variables_group.add_argument('-maxiter', type=int, default=Configuration['maxiter'], help='Maximum number of pseudos to evaluate during the optimization (soft limit). Defaults to '+str(Configuration['maxiter']))
    op_variables_group.add_argument('--rs_min', type=float, default=Configuration['bounds'][0][0], help='Lower bound for rs optimization. Defaults to '+str(Configuration['bounds'][0][0]))
    op_variables_group.add_argument('--rs_max', type=float, default=Configuration['bounds'][0][1], help='Upper bound for rs optimization. Defaults to '+str(Configuration['bounds'][0][1]))
    op_variables_group.add_argument('--rp_min', type=float, default=Configuration['bounds'][1][0], help='Lower bound for rp optimization. Defaults to '+str(Configuration['bounds'][1][0]))
    op_variables_group.add_argument('--rp_max', type=float, default=Configuration['bounds'][1][1], help='Upper bound for rp optimization. Defaults to '+str(Configuration['bounds'][1][1]))
    op_variables_group.add_argument('--rd_min', type=float, default=Configuration['bounds'][2][0], help='Lower bound for rd optimization. Defaults to '+str(Configuration['bounds'][2][0]))
    op_variables_group.add_argument('--rd_max', type=float, default=Configuration['bounds'][2][1], help='Upper bound for rd optimization. Defaults to '+str(Configuration['bounds'][2][1]))
    op_variables_group.add_argument('--rf_min', type=float, default=Configuration['bounds'][3][0], help='Lower bound for rf optimization. Defaults to '+str(Configuration['bounds'][3][0]))
    op_variables_group.add_argument('--rf_max', type=float, default=Configuration['bounds'][3][1], help='Upper bound for rf optimization. Defaults to '+str(Configuration['bounds'][3][1]))
    op_variables_group.add_argument('--rc_min', type=float, default=Configuration['bounds'][4][0], help='Lower bound for rc optimization. Defaults to '+str(Configuration['bounds'][4][0]))
    op_variables_group.add_argument('--rc_max', type=float, default=Configuration['bounds'][4][1], help='Upper bound for rc optimization. Defaults to '+str(Configuration['bounds'][4][1]))
    op_variables_group.add_argument('--no_rcore', action='store_false', help="Don't optimize the rcore variable when letting the program decide the optimization_variables automatically. Only relevant for core corrections.")

    hardness_group = parser.add_argument_group('Hardness penalization variables (b /(1.0 + exp((q0 - qmax)/a))) where qmax comes from Atom FOURIER_QMAX output')
    hardness_group.add_argument('-a', type=float, default=Configuration['a'], help='Sigmoind width. Defaults to '+str(Configuration['a']))
    hardness_group.add_argument('-b', type=float, default=Configuration['b'], help='Sigmoind max value. Defaults to '+str(Configuration['b']))
    hardness_group.add_argument('-q0', type=float, default=Configuration['q0'], help='Sigmoind change location. Defaults to '+str(Configuration['q0']))

    args = parser.parse_args()
    proces_arguments(args)

def proces_arguments(args):
    Configuration['pg'] = args.pg
    Configuration['pt'] = args.pt
    Configuration['data_dir'] = os.path.join(Configuration['script_dir'], args.data_dir)
    Configuration['data_pfile'] = os.path.join(Configuration['data_dir'], 'temporary.inp')
    Configuration['data_tfile'] = os.path.join(Configuration['data_dir'], 'test.inp')
    Configuration['out_dir'] = os.path.join(Configuration['script_dir'], args.out_dir)
    Configuration['maxiter'] = args.maxiter
    Configuration['bounds'][0][0] = args.rs_min
    Configuration['bounds'][0][1] = args.rs_max
    Configuration['bounds'][1][0] = args.rp_min
    Configuration['bounds'][1][1] = args.rp_max
    Configuration['bounds'][2][0] = args.rd_min
    Configuration['bounds'][2][1] = args.rd_max
    Configuration['bounds'][3][0] = args.rf_min
    Configuration['bounds'][3][1] = args.rf_max
    Configuration['bounds'][4][0] = args.rc_min
    Configuration['bounds'][4][1] = args.rc_max
    Configuration['a'] = args.a
    Configuration['b'] = args.b
    Configuration['q0'] = args.q0
    Configuration['pseudo_config']['r_core_flag'] = args.r_core_flag

    if args.input_psuedo is not None:
        Configuration['input_psuedo'] = os.path.join(Configuration['script_dir'], args.input_psuedo)

    if args.input_test is not None:
        Configuration['input_test'] = os.path.join(Configuration['script_dir'], args.input_test)

    # program functions
    if args.check_files:
        find_files()

    if args.results:
        configure(empty_out = False, load_config = True)
        results()

    if args.evaluate:
        configure(empty_out = False)
        if args.rs is not None:
            Configuration['pseudo_config']['rs'] = args.rs

        if args.rp is not None:
            Configuration['pseudo_config']['rp'] = args.rp

        if args.rd is not None:
            Configuration['pseudo_config']['rd'] = args.rd

        if args.rf is not None:
            Configuration['pseudo_config']['rf'] = args.rf

        if args.rcore is not None:
            Configuration['pseudo_config']['rcore'] = args.rcore

        evaluate_pseudo()
        results(evaluation = True)

    if args.optimize:
        configure(empty_out = args.resume, load_config = not args.resume)
        if args.rs is not None:
            Configuration['pseudo_config']['rs'] = args.rs

        if args.rp is not None:
            Configuration['pseudo_config']['rp'] = args.rp

        if args.rd is not None:
            Configuration['pseudo_config']['rd'] = args.rd

        if args.rf is not None:
            Configuration['pseudo_config']['rf'] = args.rf

        if args.rcore is not None:
            Configuration['pseudo_config']['rcore'] = args.rcore

        if args.optimization_variables is not None:
            if len(args.optimization_variables) != 0:
                Configuration['opvars'] = list(set(args.optimization_variables))
                Configuration['opvars'] = sorted(Configuration['opvars'], key=lambda x: ['rs', 'rp', 'rd', 'rf', 'rcore'].index(x))

        if len(Configuration['opvars']) == 0:
            for orbital in Configuration['pseudo_config']['orbital_block']:
                Configuration['opvars'].append(Orbital_Keys[orbital[1]])
            if Configuration['pseudo_config']['run_type'] == 'pe' and args.no_rcore:
                Configuration['opvars'].append('rcore')

            Configuration['opvars'] = sorted(Configuration['opvars'], key=lambda x: ['rs', 'rp', 'rd', 'rf', 'rcore'].index(x))

        res = optimize.dual_annealing(function_to_optimize, [Configuration['bounds'][Orbital_Keys_Rev[key]] for key in Configuration['opvars']], maxfun = Configuration['maxiter'], maxiter = int(1e5), no_local_search=True)
        print(res)
        results()

def configure(empty_out = True, load_config = False):
    global Configuration
    find_files()
    read_pseudo(Configuration['input_psuedo'])

    Configuration['tests'] = read_test(Configuration['input_test'])

    #If Data Doesn't Exist, Create It, Else, Format It
    if not os.path.exists(Configuration['data_dir']):
        os.mkdir(Configuration['data_dir'])
        print("directory " , Configuration['data_dir'] ,  " created ")
    empty_dir(Configuration['data_dir'])

    #If Out Doesn't Exist, Create It, Else, Format It
    if not os.path.exists(Configuration['out_dir']):
        os.mkdir(Configuration['out_dir'])
        print("directory " , Configuration['out_dir'] ,  " created ")

    if empty_out:
        empty_dir(Configuration['out_dir'])
        # Create out.log With the Header
        with open(os.path.join(Configuration['out_dir'], 'out.log'), 'w') as file:
            print('rs,rp,rd,rf,r_core_flag,rcore,DeltaE,DeltaEig,hardness,score', file = file)

    if not os.path.exists(os.path.join(Configuration['out_dir'], 'out.log')):
        with open(os.path.join(Configuration['out_dir'], 'out.log'), 'w') as file:
            print('rs,rp,rd,rf,r_core_flag,rcore,DeltaE,DeltaEig,hardness,score', file = file)

    if load_config:
        if os.path.exists(os.path.join(Configuration['out_dir'], 'config.json')):
            print('loading config.json.')
            with open(os.path.join(Configuration['out_dir'], 'config.json'), 'r') as config_file:
                Configuration = json.load(config_file)
        else:
            print('config.json not found. Using command line variables.')

    write_test(Configuration['tests'], file = Configuration['data_tfile'])
    write_test(Configuration['tests'], file = os.path.join(Configuration['out_dir'], os.path.basename(Configuration['input_test'])))

    write_pseudo(file = Configuration['data_pfile'])
    write_pseudo(file = os.path.join(Configuration['out_dir'], 'original-' + os.path.basename(Configuration['input_psuedo'])))

def find_files():
    #Serch for Transference Test File
    if Configuration['input_test'] == '':
        for file in glob.glob(os.path.join(Configuration['script_dir'], '*test*')):
            Configuration['input_test'] = file

    if Configuration['input_test'] != '' and os.path.isfile(Configuration['input_test']):
        print('Tests file found: ', Configuration['input_test'])
    else:
        print('Tests file not found')
        exit()
    #Serch for Pseudo File
    if Configuration['input_psuedo'] == '':
        for file in os.listdir(Configuration['script_dir']):
            if file.endswith(".inp") and os.path.join(Configuration['script_dir'], file) != Configuration['input_test']:
                Configuration['input_psuedo'] = os.path.join(Configuration['script_dir'], file)

    if Configuration['input_psuedo'] != '' and os.path.isfile(Configuration['input_psuedo']):
        print('Pseudopotential file found: ', Configuration['input_psuedo'])
    else:
        print('Pseudopotential file not found')
        exit()

    #Search for ATOM Scripts
    if os.path.isfile(Configuration['pg']):
        print('pg.sh found: ', Configuration['pg'])
    else:
        print('pg.sh not found, edit script and correct path.')
        exit()

    if os.path.isfile(Configuration['pt']):
        print('pt.sh found: ', Configuration['pt'])
    else:
        print('pt.sh not found, edit script and correct path.')
        exit()

def function_to_optimize(x):
    for i, key in enumerate(Configuration['opvars']):
        Configuration['pseudo_config'][key] = x[i]
    return evaluate_pseudo()

def results(evaluation=False):
    if not os.path.exists(Configuration['out_dir']):
        print('There\'s no Out directory')
        exit()

    with open(os.path.join(Configuration['out_dir'], 'config.json'), 'w') as config_file:
        json.dump(Configuration, config_file, indent=4)

    if not os.path.exists(os.path.join(Configuration['out_dir'], 'out.log')):
        print('There\'s no out.log file')
        exit()

    num_lines = sum(1 for line in open(os.path.join(Configuration['out_dir'], 'out.log')))
    if num_lines < 2:
        print('There are no results in Out directory')
        exit()

    df = pd.read_csv(os.path.join(Configuration['out_dir'], 'out.log'))
    min_score = df['score'].min()
    entries_with_min_score = df[df['score'] == min_score].drop_duplicates()
    entries_with_min_score.to_csv(os.path.join(Configuration['out_dir'], 'best_entries.log'), index=False, float_format='%.5f')

    first_entry = entries_with_min_score.iloc[0]
    x = [first_entry['rs'], first_entry['rp'], first_entry['rd'], first_entry['rf'], first_entry['rcore']]
    print_pseudo()
    print('Best pseudos at:')
    print(entries_with_min_score.applymap(lambda x: f"{x:.5f}"))

    if evaluation:
        last_row_df = df.iloc[[-1]]
        print('Evaluated pseudo:')
        print(last_row_df.applymap(lambda x: f"{x:.5f}"))

    if not evaluation:
        function_to_optimize(x)

    plots()

def copy_files(src_pattern, dst_dir):
    for filepath in glob.glob(src_pattern):
        shutil.copy(filepath, dst_dir)

def plots():
    os.chdir(os.path.join(Configuration['data_dir'], 'test-temporary'))
    os.system('gnuplot charge.gps vcharge.gps vspin.gps pt.gps')
    os.chdir(os.path.join(Configuration['data_dir'], 'temporary'))
    os.system('gnuplot pseudo.gps')
    os.chdir(Configuration['script_dir'])
    copy_files(os.path.join(Configuration['data_dir'], 'test-temporary', '*.ps'), Configuration['out_dir'])
    copy_files(os.path.join(Configuration['data_dir'], 'temporary', '*.ps'), Configuration['out_dir'])

    # Move Pseudos and Plotst to Out Directory
    write_pseudo(file = os.path.join(Configuration['out_dir'], os.path.basename(Configuration['input_psuedo'])))
    shutil.copy(os.path.join(Configuration['data_dir'], 'temporary.psf'), os.path.join(Configuration['out_dir'], Configuration['pseudo_config']['chemical_symbol'].strip() + '.psf'))
    shutil.copy(os.path.join(Configuration['data_dir'], 'temporary.psml'), os.path.join(Configuration['out_dir'], Configuration['pseudo_config']['chemical_symbol'].strip() + '.psml'))
    shutil.copy(os.path.join(Configuration['data_dir'], 'temporary.vps'), os.path.join(Configuration['out_dir'], Configuration['pseudo_config']['chemical_symbol'].strip() + '.vps'))

def extract_eigen_val(lines):
    eig_ae = [0.0]
    eig_pt = [0.0]
    is_eig_pt_section = False

    for line in lines:
        if 'End' in line:
            is_eig_pt_section = True
            continue  # Skip the line containing 'End'

        for num in line.split()[:-3]: # only first eigenvalue
            # Check if num is a number (float or int)
            try:
                val = float(num)
                if is_eig_pt_section:
                    eig_pt.append(val)
                else:
                    eig_ae.append(val)
            except ValueError:
                continue

    min_len = min(len(eig_ae), len(eig_pt))
    return sum([abs(eig_ae[i] - eig_pt[i]) for i in range(min_len)])

def eigen_val_diff():
    os.system("grep '&v' test-temporary/OUT | grep s > s.txt")
    os.system("grep '&v' test-temporary/OUT | grep p > p.txt")
    os.system("grep '&v' test-temporary/OUT | grep d > d.txt")
    os.system("grep '&v' test-temporary/OUT | grep f > f.txt")

    lines = []
    with open('s.txt', 'r') as file:
        lines = file.readlines()

    eig_s = extract_eigen_val(lines)

    lines = []
    with open('p.txt', 'r') as file:
        lines = file.readlines()

    eig_p = extract_eigen_val(lines)

    lines = []
    with open('d.txt', 'r') as file:
        lines = file.readlines()

    eig_d = extract_eigen_val(lines)

    lines = []
    with open('f.txt', 'r') as file:
        lines = file.readlines()

    eig_f = extract_eigen_val(lines)

    return eig_s + eig_p + eig_d + eig_f

def evaluate_pseudo():
    global Count
    Count += 1

    write_pseudo(file = Configuration['data_pfile'])
    empty_dir(Configuration['data_dir'], files_to_keep = [Configuration['data_pfile'], Configuration['data_tfile']])

    os.chdir(Configuration['data_dir'])
    os.system('sh ' + Configuration['pg'] + ' temporary.inp >/dev/null 2>&1')

    # If Fails Leave the Function
    if not os.path.isfile(os.path.join(Configuration['data_dir'], 'temporary.vps')):
        os.chdir(Configuration['script_dir']) # Return to Top Directory
        print(str(Count) + 'th Pseudo failed')
        return Bad_Return

    # If Fails Leave the Function
    if not os.path.isfile(os.path.join('temporary', 'FOURIER_QMAX')):
        os.chdir(Configuration['script_dir']) # Return to Top Directory
        print(str(Count) + 'th Pseudo failed')
        return Bad_Return

    # Execute Transferability Tests
    os.system('sh ' + Configuration['pt'] + ' test.inp temporary.vps >/dev/null 2>&1')

    # If Fails Leave the Function
    if not os.path.isfile(os.path.join('test-temporary', 'ECONF_DIFFS')):
        os.chdir(Configuration['script_dir']) # Return to Top Directory
        print(str(Count) + 'th Test failed')
        return Bad_Return

    # If Fails Leave the Function
    if not os.path.isfile(os.path.join('test-temporary', 'OUT')):
        os.chdir(Configuration['script_dir']) # Return to Top Directory
        print(str(Count) + 'th Test failed')
        return Bad_Return

    DeltaEig = eigen_val_diff()
    DeltaE = 0

    with open(os.path.join('test-temporary', 'ECONF_DIFFS'), 'r') as file:
        next(file)
        for line in file:
            DeltaE += sum([abs(float(x)) for x in line.split()])

    qmax = 0.0
    with open(os.path.join('temporary', 'FOURIER_QMAX'), 'r') as file:
        file.readline()
        qmax = sum([hardness_penalization(float(x)) for x in file.readline().split()])

    report = [Configuration['pseudo_config']['rs'], Configuration['pseudo_config']['rp'], Configuration['pseudo_config']['rd'], Configuration['pseudo_config']['rf'], Configuration['pseudo_config']['r_core_flag'], Configuration['pseudo_config']['rcore'], DeltaE, DeltaEig, qmax, DeltaE + DeltaEig + qmax]
    with open(os.path.join(Configuration['out_dir'], 'out.log'), 'a') as file:
        print(','.join([f"{num:.5f}" for num in report[:-1]] + [f"{report[-1]:.8f}"]), file=file)

    return DeltaE + DeltaEig + qmax

def empty_dir(directory, files_to_keep = []):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path not in files_to_keep and filename not in files_to_keep:
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def read_test(file):
    tests = []
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].split('#')[0]

    lines = [line for line in lines if not line.isspace()]
    lines = [line.rstrip() for line in lines]
    lines = [line for line in lines if line.strip()]

    for i, line in enumerate(lines):
        if line.split()[0][-2:] == 'pt':
            lines = lines[:i]
            break

    number_tests = 0
    test_locations = []
    for i, line in enumerate(lines):
        if line.split()[0][-2:] == 'ae':
            number_tests += 1
            test_locations.append(i)
    test_locations.append(len(lines))

    for i in range(number_tests):
        tests.append(read_ae(lines[test_locations[i]:test_locations[i + 1]]))

    return tests

def print_test(tests):
    for ae_config in tests:
        print_ae(ae_config)

    print('#')
    print('# Pseudopotential test calculations')
    print('#')

    for ae_config in tests:
        ae_config['run_type'] = 'pt'
        print_ae(ae_config)
        ae_config['run_type'] = 'ae'

def write_test(tests, file = 'test.inp'):
    output = []
    for ae_config in tests:
        write_ae(ae_config, output)

    output.append('#')
    output.append('# Pseudopotential test calculations')
    output.append('#')

    for ae_config in tests:
        ae_config['run_type'] = 'pt'
        write_ae(ae_config, output)
        ae_config['run_type'] = 'ae'

    #Join all parts with newlines and write to the file
    with open(file, 'w') as f:
        f.write("\n".join(output))

def check_ae(ae_config):
    error = False
    if ae_config['run_type'] not in ['ae', 'pt']:
        error = True
        print('run_type should be ae or pt, got: ' + ae_config['run_type'])

    if ae_config['XC_flavor'] not in ['xc', 'wi', 'hl', 'gl', 'bh', 'ca', 'pw', 'pb', 'wp', 'rp', 'rv', 'ps', 'wc', 'jo', 'jh', 'go', 'gh', 'am', 'bl', 'vw', 'vl', 'vk', 'vc', 'vb', 'vv']:
        error = True
        print('XC_flavor should be xc, wi, hl, gl, bh, ca, pw, pb, wp, rp, rv, ps, wc, jo, jh, go, gh, am, bl, vw, vl, vk, vc, vb or vv, got: ' + ae_config['XC_flavor'])

    if ae_config['XC_flavor'] == 'xc' and (not len(ae_config['XC_id']) == 8 or not ae_config['XC_id'][0] == '0' or not ae_config['XC_id'][4] == '0'):
        error = True
        print('XC_id has to be a 8 digit integer that follows 0XXX0YYY where XXX and YYY are libxc IDs. XXX is exchange and YYY is correlation. Got: ' + ae_config['XC_id'])

    if error:
        raise ValueError('Input has errors')

def read_ae(lines):
    ae_config = {'XC_id' : '00000000', 'spin_relativistic': ''}
    # first line
    ae_config['run_type'] = lines[0].split()[0][:2]
    ae_config['title'] = lines[0][lines[0].find(ae_config['run_type']) + 2:]
    # second line
    ae_config['chemical_symbol'] = lines[1].split()[0][-2:]
    if lines[1].split()[1][-1] in ['r', 's']:
        ae_config['spin_relativistic'] = lines[1].split()[1][-1]
        ae_config['XC_flavor'] = lines[1].split()[1][-3:-1]
    else:
        ae_config['XC_flavor'] = lines[1].split()[1][-2:]

    if ae_config['XC_flavor'] == 'xc':
        ae_config['XC_id'] = lines[1].split()[2]
    # third line
    ae_config['fractional_atomic_number'] = float(lines[2].split()[0])
    # fourth line
    ae_config['norbs_core'] = int(lines[3].split()[0])
    ae_config['norbs_valence'] = int(lines[3].split()[1])
    # orbitals block
    ae_config['orbital_block'] = []
    for i in range(1, ae_config['norbs_valence'] + 1):
        orbitals = lines[3+i].split()
        if len(orbitals) > 3:
            orbitals = [int(orbitals[0]), int(orbitals[1]), float(orbitals[2]), float(orbitals[3])]
        else:
            orbitals = [int(orbitals[0]), int(orbitals[1]), float(orbitals[2]), 0.0]

        ae_config['orbital_block'].append(orbitals)

    check_ae(ae_config)
    return ae_config

def print_ae(ae_config):
    # (format 3x,a2,a50)
    print(f"{'':3}{ae_config['run_type']:2}{'':1}{ae_config['title']:49}")
    # (format 3x,a2,3x,a2,a1,1x,i8)
    print(f"{'':3}{ae_config['chemical_symbol']:2}{'':3}{ae_config['XC_flavor']:2}{ae_config['spin_relativistic']:1}{'':1}{ae_config['XC_id']:8}")
    # (format f10.5)
    print(f"{ae_config['fractional_atomic_number']:10.1f}")
    # (format 2i5)
    print(f"{ae_config['norbs_core']:5d}{ae_config['norbs_valence']:5d}")
    # block of orbitals (format 2i5,2f10.3)
    for line in ae_config['orbital_block']:
        print(f"{line[0]:5d}{line[1]:5d}{line[2]:10.3f}{line[3]:10.3f}")
    print('')

def write_ae(ae_config, output = []):
    # (format 3x,a2,a50)
    output.append(f"{'':3}{ae_config['run_type']:2}{'':1}{ae_config['title']:49}")
    # (format 3x,a2,3x,a2,a1,1x,i8)
    output.append(f"{'':3}{ae_config['chemical_symbol']:2}{'':3}{ae_config['XC_flavor']:2}{ae_config['spin_relativistic']:1}{'':1}{ae_config['XC_id']:8}")
    # (format f10.5)
    output.append(f"{ae_config['fractional_atomic_number']:10.1f}")
    # (format 2i5)
    output.append(f"{ae_config['norbs_core']:5d}{ae_config['norbs_valence']:5d}")
    # block of orbitals (format 2i5,2f10.3)
    for line in ae_config['orbital_block']:
        output.append(f"{line[0]:5d}{line[1]:5d}{line[2]:10.3f}{line[3]:10.3f}")

    return output

def check_pseudo(pseudo_config = Configuration['pseudo_config']):
    error = False
    if pseudo_config['run_type'] not in ['pg', 'pe']:
        error = True
        print('run_type should be pg or pe, got: ' + pseudo_config['run_type'])

    if pseudo_config['PS_flavor'] not in ['hsc', 'ker', 'tm2']:
        error = True
        print('PS_flavor should be hsc, ker or tm2, got: ' + pseudo_config['PS_flavor'])

    if pseudo_config['XC_flavor'] not in ['xc', 'wi', 'hl', 'gl', 'bh', 'ca', 'pw', 'pb', 'wp', 'rp', 'rv', 'ps', 'wc', 'jo', 'jh', 'go', 'gh', 'am', 'bl', 'vw', 'vl', 'vk', 'vc', 'vb', 'vv']:
        error = True
        print('XC_flavor should be xc, wi, hl, gl, bh, ca, pw, pb, wp, rp, rv, ps, wc, jo, jh, go, gh, am, bl, vw, vl, vk, vc, vb or vv, got: ' + pseudo_config['XC_flavor'])


    if pseudo_config['XC_flavor'] == 'xc' and (not len(pseudo_config['XC_id']) == 8 or not pseudo_config['XC_id'][0] == '0' or not pseudo_config['XC_id'][4] == '0'):
        error = True
        print('XC_id has to be a 8 digit integer that follows 0XXX0YYY where XXX and YYY are libxc IDs. XXX is correlation and YYY is exchange. got: ' + pseudo_config['XC_id'])

    if error:
        raise ValueError('Input has errors')

def read_pseudo(file):
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].split('#')[0]

    lines = [line for line in lines if not line.isspace()]
    lines = [line.rstrip() for line in lines]
    lines = [line for line in lines if line.strip()]
    # first line
    Configuration['pseudo_config']['run_type'] = lines[0].split()[0][:2]
    Configuration['pseudo_config']['title'] = lines[0][lines[0].find(Configuration['pseudo_config']['run_type']) + 2:]
    # second line
    Configuration['pseudo_config']['PS_flavor'] =  lines[1].split()[0]
    Configuration['pseudo_config']['logder_r'] =  float(lines[1].split()[1])
    # third line
    Configuration['pseudo_config']['chemical_symbol'] = lines[2].split()[0][-2:]
    if lines[2].split()[1][-1] in ['r', 's']:
        Configuration['pseudo_config']['spin_relativistic'] = lines[2].split()[1][-1]
        Configuration['pseudo_config']['XC_flavor'] = lines[2].split()[1][-3:-1]
    else:
        Configuration['pseudo_config']['XC_flavor'] = lines[2].split()[1][-2:]

    if Configuration['pseudo_config']['XC_flavor'] == 'xc':
        Configuration['pseudo_config']['XC_id'] = lines[2].split()[2]
    # fourth line
    Configuration['pseudo_config']['fractional_atomic_number'] = float(lines[3].split()[0])
    # fifth line
    Configuration['pseudo_config']['norbs_core'] = int(lines[4].split()[0])
    Configuration['pseudo_config']['norbs_valence'] = int(lines[4].split()[1])
    # orbitals block
    Configuration['pseudo_config']['orbital_block'] = []
    for i in range(1, Configuration['pseudo_config']['norbs_valence'] + 1):
        orbitals = lines[4+i].split()
        if len(orbitals) > 3:
            orbitals = [int(orbitals[0]), int(orbitals[1]), float(orbitals[2]), float(orbitals[3])]
        else:
            orbitals = [int(orbitals[0]), int(orbitals[1]), float(orbitals[2]), 0.0]

        Configuration['pseudo_config']['orbital_block'].append(orbitals)
    # last line
    i = Configuration['pseudo_config']['norbs_valence'] + 5
    if len(lines[i].split()) > 5:
        cutoff_radii = lines[i].split()[:6]
    elif len(lines[i].split()) > 3:
        cutoff_radii = lines[i].split()[:4]

    cutoff_radii = [float(x) for x in cutoff_radii]
    Configuration['pseudo_config']['rs'] = cutoff_radii[0]
    Configuration['pseudo_config']['rp'] = cutoff_radii[1]
    Configuration['pseudo_config']['rd'] = cutoff_radii[2]
    Configuration['pseudo_config']['rf'] = cutoff_radii[3]

    if len(cutoff_radii) > 5:
        Configuration['pseudo_config']['rcore'] = cutoff_radii[5]

    check_pseudo(Configuration['pseudo_config'])

def print_pseudo():
    # (format 3x,a2,a50)
    print(f"{'':3}{Configuration['pseudo_config']['run_type']:2}{'':1}{Configuration['pseudo_config']['title']:49}")
    # (format 8x, a3, f9.3)
    print(f"{'':8}{Configuration['pseudo_config']['PS_flavor']:3}{Configuration['pseudo_config']['logder_r']:9.3f}")
    # (format 3x,a2,3x,a2,a1,1x,i8)
    print(f"{'':3}{Configuration['pseudo_config']['chemical_symbol']:2}{'':3}{Configuration['pseudo_config']['XC_flavor']:2}{Configuration['pseudo_config']['spin_relativistic']:1}{'':1}{Configuration['pseudo_config']['XC_id']:8}")
    # (format f10.5)
    print(f"{Configuration['pseudo_config']['fractional_atomic_number']:10.1f}")
    # (format 2i5)
    print(f"{Configuration['pseudo_config']['norbs_core']:5d}{Configuration['pseudo_config']['norbs_valence']:5d}")
    # block of orbitals (format 2i5,2f10.3)
    for line in Configuration['pseudo_config']['orbital_block']:
        print(f"{line[0]:5d}{line[1]:5d}{line[2]:10.3f}{line[3]:10.3f}")

    # (format 6f10.5)
    print(f"{Configuration['pseudo_config']['rs']:10.5f}{Configuration['pseudo_config']['rp']:10.5f}{Configuration['pseudo_config']['rd']:10.5f}{Configuration['pseudo_config']['rf']:10.5f}{Configuration['pseudo_config']['r_core_flag']:10.5f}{Configuration['pseudo_config']['rcore']:10.5f}")
    print('')

def write_pseudo(file = 'temporary.inp'):
    output = []
    # (format 3x,a2,a50)
    output.append(f"{'':3}{Configuration['pseudo_config']['run_type']:2}{'':1}{Configuration['pseudo_config']['title']:49}")
    # (format 8x, a3, f9.3)
    output.append(f"{'':8}{Configuration['pseudo_config']['PS_flavor']:3}{Configuration['pseudo_config']['logder_r']:9.3f}")
    # (format 3x,a2,3x,a2,a1,1x,i8)
    output.append(f"{'':3}{Configuration['pseudo_config']['chemical_symbol']:2}{'':3}{Configuration['pseudo_config']['XC_flavor']:2}{Configuration['pseudo_config']['spin_relativistic']:1}{'':1}{Configuration['pseudo_config']['XC_id']:8}")
    # (format f10.5)
    output.append(f"{Configuration['pseudo_config']['fractional_atomic_number']:10.1f}")
    # (format 2i5)
    output.append(f"{Configuration['pseudo_config']['norbs_core']:5d}{Configuration['pseudo_config']['norbs_valence']:5d}")
    # block of orbitals (format 2i5,2f10.3)
    for line in Configuration['pseudo_config']['orbital_block']:
        output.append(f"{line[0]:5d}{line[1]:5d}{line[2]:10.3f}{line[3]:10.3f}")
    # (format 6f10.5)
    output.append(f"{Configuration['pseudo_config']['rs']:10.5f}{Configuration['pseudo_config']['rp']:10.5f}{Configuration['pseudo_config']['rd']:10.5f}{Configuration['pseudo_config']['rf']:10.5f}{Configuration['pseudo_config']['r_core_flag']:10.5f}{Configuration['pseudo_config']['rcore']:10.5f}")

    # Join all parts with newlines and write to the file
    with open(file, 'w') as f:
        f.write("\n".join(output))

if __name__ == '__main__':
    main()

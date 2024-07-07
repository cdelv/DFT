#########################################################################
# @Copyright Carlos Andrés del Valle. cdelv@unal.edu.co. Jul 4, 2024.
#########################################################################
import glob
import os
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy import optimize

# maximun number of pseudos created in the optimization (soft limit)
Maxiter = 1500

# Bounds for the cutoff radii
r_core_flag = 1.0
Bounds = [
    [0, 2.5], # S
    [0, 3.5], # P
    [0, 4],   # D
    [0, 4.5],   # F
    [0, 3]    # rcore
]
Bounds = np.array(Bounds).astype(np.float64)

# Hardness Penalization Parameters using atom FOURIER_QMAX
a = 0.5
b = 10.0
q0 = 17.0
Bad_Return = 10*b
def hardness_penalization(qmax):
    return b/(1.0+np.exp((q0-qmax)/a))

Script_Dir = os.path.abspath(os.getcwd())
pg="/opt/Atom/atom-4.2.7-100/Tutorial/Utils/pg.sh"
pt="/opt/Atom/atom-4.2.7-100/Tutorial/Utils/pt.sh"
Pfile=''
Tfile=''
Data_Dir = ''
Out_Dir = ''
Data_Pfile = ''
Data_Tfile = ''
Count = 0

Pseudo_Config = {
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
    'rcore_flag': r_core_flag,
    'rcore': 0.0,
    'orbital_block': [[1, 0, 0.5, 0]]
}
Orbital_Keys = {0 : 'rs', 1 : 'rp', 2 : 'rd', 3 : 'rf', 4 : 'rcore'}
Orbital_Keys_Rev =  {'rs' : 0, 'rp' : 1, 'rd' : 2, 'rf' : 3, 'rcore' : 4}
Key_Order = ['rs', 'rp', 'rd', 'rf', 'rcore']
Keys = []
Tests = []

def main():
    parser = argparse.ArgumentParser(
        prog='',
        description='This program uses the annealing optimization algorithm to find the best combination of cutoff radii to create the pseudopotential. It uses the transferability test to evaluate how "good" a pseudopotential is. To use it, you need to provide the path of the Atom pg.sh and pt.sh scripts and have a pseudo input file and a transferability test file. Read the rest of the flags for details.',
        epilog='''
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
    conf_group.add_argument('-pg', type=str, default=pg, help='Path to pg.sh Atom script. Defaults to '+pg)
    conf_group.add_argument('-pt', type=str, default=pt, help='Path to pt.sh Atom script. Defaults to '+pt)
    conf_group.add_argument('-o', '--out_dir', type=str, default='Out', help='Name of the output directory. Here the optimization log and results will be stored. Defaults to Out')
    conf_group.add_argument('-d', '--data_dir', type=str, default='Data', help='Name of the data directory. Here the pseudos are generated and hold temporary files. It gets ovewritten each time a new pseudo is created. Defaults to Data')
    conf_group.add_argument('-rcf', '--r_core_flag', type=float, default=r_core_flag, help='Value of the r_core_flag to use. The program does not optimize this value. Defaults to '+str(r_core_flag))

    variables_group = parser.add_argument_group('Program evaluation variables')
    variables_group.add_argument('-rs', type=float, help='Value of the s orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rp', type=float, help='Value of the p orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rd', type=float, help='Value of the d orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rf', type=float, help='Value of the f orbital cutoff. If set, overrides the value present in the input pseudo *.inp file.')
    variables_group.add_argument('-rc', '--rcore', type=float, help='Value of the rcore cutoff. If set in an evaluation run, overrides the value present in the input pseudo *.inp file. Only used in corre correction runs (pe instead of pg)')

    op_variables_group = parser.add_argument_group('Program optimization variables')
    op_variables_group.add_argument('--resume', action='store_false', help='Perform an optimization run without deleting previous results.')
    op_variables_group.add_argument('-opvar', '--optimization_variables', nargs='*', type=str, choices=['rs', 'rp', 'rd', 'rf', 'rcore'], help="Optimize the pseudopotential using only the selected orbitals' cutoff radius. Defaults to the orbitals detected in the pseudo. For the fixed orbitals, it uses the value in the input *.inp pseudo file or the value set by the -r* flags. If core corrections, the progrma includes rc in the default orbitals to optimize.")
    op_variables_group.add_argument('-maxiter', type=int, default=Maxiter, help='Maximum number of pseudos to evaluate during the optimization (soft limit). Defaults to '+str(Maxiter))
    op_variables_group.add_argument('--rs_min', type=float, default=Bounds[0][0], help='Lower bound for rs optimization. Defaults to '+str(Bounds[0][0]))
    op_variables_group.add_argument('--rs_max', type=float, default=Bounds[0][1], help='Upper bound for rs optimization. Defaults to '+str(Bounds[0][1]))
    op_variables_group.add_argument('--rp_min', type=float, default=Bounds[1][0], help='Lower bound for rp optimization. Defaults to '+str(Bounds[1][0]))
    op_variables_group.add_argument('--rp_max', type=float, default=Bounds[1][1], help='Upper bound for rp optimization. Defaults to '+str(Bounds[1][1]))
    op_variables_group.add_argument('--rd_min', type=float, default=Bounds[2][0], help='Lower bound for rd optimization. Defaults to '+str(Bounds[2][0]))
    op_variables_group.add_argument('--rd_max', type=float, default=Bounds[2][1], help='Upper bound for rd optimization. Defaults to '+str(Bounds[2][1]))
    op_variables_group.add_argument('--rf_min', type=float, default=Bounds[3][0], help='Lower bound for rf optimization. Defaults to '+str(Bounds[3][0]))
    op_variables_group.add_argument('--rf_max', type=float, default=Bounds[3][1], help='Upper bound for rf optimization. Defaults to '+str(Bounds[3][1]))
    op_variables_group.add_argument('--rc_min', type=float, default=Bounds[4][0], help='Lower bound for rc optimization. Defaults to '+str(Bounds[4][0]))
    op_variables_group.add_argument('--rc_max', type=float, default=Bounds[4][1], help='Upper bound for rc optimization. Defaults to '+str(Bounds[4][1]))
    op_variables_group.add_argument('--no_rcore', action='store_false', help="Don't optimize the rcore variable when letting the program decide the optimization_variables automatically. Only relevant for core corrections.")

    args = parser.parse_args()
    proces_arguments(args)

def proces_arguments(args):
    # Change parsed values
    global Pfile
    if args.input_psuedo is not None:
        Pfile = os.path.join(Script_Dir, args.input_psuedo)
        
    global Tfile
    if args.input_test is not None:
        Tfile = os.path.join(Script_Dir, args.input_test)

    global pg
    pg = args.pg

    global pt
    pt = args.pt

    global Data_Dir
    Data_Dir = os.path.join(Script_Dir, args.data_dir)
    global Data_Pfile
    Data_Pfile = os.path.join(Data_Dir, 'temporary.inp')
    global Data_Tfile
    Data_Tfile = os.path.join(Data_Dir, 'test.inp')

    global Out_Dir
    Out_Dir = os.path.join(Script_Dir, args.out_dir)

    global r_core_flag
    r_core_flag = args.r_core_flag

    global Bounds
    Bounds[0][0] = args.rs_min
    Bounds[0][1] = args.rs_max
    Bounds[1][0] = args.rp_min
    Bounds[1][1] = args.rp_max
    Bounds[2][0] = args.rd_min
    Bounds[2][1] = args.rd_max
    Bounds[3][0] = args.rf_min
    Bounds[3][1] = args.rf_max
    Bounds[4][0] = args.rc_min
    Bounds[4][1] = args.rc_max

    global Maxiter
    Maxiter = args.maxiter

    if args.check_files:
        find_files()

    if args.results:
        configure(empty_out = False)
        results()

    if args.evaluate:
        configure(empty_out = False)
        if args.rs is not None:
            Pseudo_Config['rs'] = args.rs 

        if args.rp is not None:
            Pseudo_Config['rp'] = args.rp

        if args.rd is not None:
            Pseudo_Config['rd'] = args.rd

        if args.rf is not None:
            Pseudo_Config['rf'] = args.rf

        if args.rcore is not None:
            Pseudo_Config['rcore'] = args.rcore

        evaluate_pseudo()
        results(evaluation = True)

    if args.optimize:
        configure(empty_out = args.resume)
        if args.rs is not None:
            Pseudo_Config['rs'] = args.rs 

        if args.rp is not None:
            Pseudo_Config['rp'] = args.rp

        if args.rd is not None:
            Pseudo_Config['rd'] = args.rd

        if args.rf is not None:
            Pseudo_Config['rf'] = args.rf

        if args.rcore is not None:
            Pseudo_Config['rcore'] = args.rcore

        global Keys
        if args.optimization_variables is not None:
            if len(args.optimization_variables) != 0:
                Keys = list(set(args.optimization_variables))
                Keys = sorted(Keys, key=lambda x: Key_Order.index(x))
        
        if len(Keys) == 0:
            for orbital in Pseudo_Config['orbital_block']:
                Keys.append(Orbital_Keys[orbital[1]])
            if Pseudo_Config['run_type'] == 'pe' and args.no_rcore:
                Keys.append('rcore')

            Keys = sorted(Keys, key=lambda x: Key_Order.index(x))

        res = optimize.dual_annealing(function_to_optimize, [Bounds[Orbital_Keys_Rev[key]] for key in Keys], maxfun = Maxiter, maxiter = int(1e5), no_local_search=True)
        print(res)
        results()

def function_to_optimize(x):
    for i, key in enumerate(Keys):
        Pseudo_Config[key] = x[i]

    return evaluate_pseudo()

def results(evaluation=False):
    if not os.path.exists(Out_Dir):
        print('There\'s no Out directory')
        exit()

    if not os.path.exists(os.path.join(Out_Dir, 'out.log')):
        print('There\'s no out.log file')
        exit()

    num_lines = sum(1 for line in open(os.path.join(Out_Dir, 'out.log')))
    if num_lines < 2:
        print('There are no results in Out directory')
        exit()

    df = pd.read_csv(os.path.join(Out_Dir, 'out.log'))
    min_score = df['score'].min()
    entries_with_min_score = df[df['score'] == min_score].drop_duplicates()
    entries_with_min_score.to_csv(os.path.join(Out_Dir, 'best_entries.log'), index=False, float_format='%.5f')
    
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
    os.chdir(os.path.join(Data_Dir, 'test-temporary'))
    os.system('gnuplot charge.gps vcharge.gps vspin.gps pt.gps')
    os.chdir(os.path.join(Data_Dir, 'temporary'))
    os.system('gnuplot pseudo.gps')
    os.chdir(Script_Dir)
    copy_files(os.path.join(Data_Dir, 'test-temporary', '*.ps'), Out_Dir)
    copy_files(os.path.join(Data_Dir, 'temporary', '*.ps'), Out_Dir)

    # Move Pseudos and Plotst to Out Directory
    write_pseudo(file=os.path.join(Out_Dir, os.path.basename(Pfile)))
    shutil.copy(os.path.join(Data_Dir, 'temporary.psf'), os.path.join(Out_Dir, Pseudo_Config['chemical_symbol'].strip()+'.psf'))
    shutil.copy(os.path.join(Data_Dir, 'temporary.psml'), os.path.join(Out_Dir, Pseudo_Config['chemical_symbol'].strip()+'.psml'))
    shutil.copy(os.path.join(Data_Dir, 'temporary.vps'), os.path.join(Out_Dir, Pseudo_Config['chemical_symbol'].strip()+'.vps'))

def extract_eigen_val(lines):
    eig_ae = [0]
    eig_pt = [0]
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

    write_pseudo(file=Data_Pfile)
    empty_dir(Data_Dir, files_to_keep = [Data_Pfile, Data_Tfile])

    os.chdir(Data_Dir)
    os.system('sh ' + pg + ' temporary.inp >/dev/null 2>&1')

    # If Fails Leave the Function 
    if not os.path.isfile(os.path.join(Data_Dir, 'temporary.vps')):
        os.chdir(Script_Dir) # Return to Top Directory
        print(str(Count)+'th Pseudo failed')
        return Bad_Return

    # If Fails Leave the Function 
    if not os.path.isfile(os.path.join('temporary', 'FOURIER_QMAX')):
        os.chdir(Script_Dir) # Return to Top Directory
        print(str(Count)+'th Pseudo failed')
        return Bad_Return

    # Execute Transferability Tests
    os.system('sh ' + pt + ' test.inp temporary.vps >/dev/null 2>&1')

    # If Fails Leave the Function 
    if not os.path.isfile(os.path.join('test-temporary', 'ECONF_DIFFS')):
        os.chdir(Script_Dir) # Return to Top Directory
        print(str(Count)+'th Test failed')
        return Bad_Return

    # If Fails Leave the Function 
    if not os.path.isfile(os.path.join('test-temporary', 'OUT')):
        os.chdir(Script_Dir) # Return to Top Directory
        print(str(Count)+'th Test failed')
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

    report = [Pseudo_Config['rs'], Pseudo_Config['rp'], Pseudo_Config['rd'], Pseudo_Config['rf'], Pseudo_Config['rcore_flag'], Pseudo_Config['rcore'], DeltaE, DeltaEig, qmax, DeltaE + DeltaEig + qmax]
    with open(os.path.join(Out_Dir, 'out.log'), 'a') as file:
        print(','.join([f"{num:.5f}" for num in report[:-1]] + [f"{report[-1]:.8f}"]), file=file)

    return DeltaE + DeltaEig + qmax

    
def configure(empty_out = True):
    find_files()
    read_pseudo(Pfile)

    global Tests
    Tests = read_test(Tfile)

    #If Data Doesn't Exist, Create It, Else, Format It
    dirName = Data_Dir
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("directory " , dirName ,  " created ")
    empty_dir(dirName)

    #If Out Doesn't Exist, Create It, Else, Format It
    dirName = Out_Dir
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("directory " , dirName ,  " created ")

    if empty_out:
        empty_dir(dirName)
        # Create out.log With the Header
        with open(os.path.join(Out_Dir, 'out.log'), 'w') as file:
            print('rs,rp,rd,rf,rcore_flag,rcore,DeltaE,DeltaEig,hardness,score', file=file)

    if not os.path.exists(os.path.join(Out_Dir, 'out.log')):
        with open(os.path.join(Out_Dir, 'out.log'), 'w') as file:
            print('rs,rp,rd,rf,rcore_flag,rcore,DeltaE,DeltaEig,hardness,score', file=file)

    write_test(Tests, file=Data_Tfile)
    write_test(Tests, file=os.path.join(Out_Dir, os.path.basename(Tfile)))

    write_pseudo(file=Data_Pfile)
    write_pseudo(file=os.path.join(Out_Dir, 'original-'+os.path.basename(Pfile)))

def find_files():
    #Serch for Transference Test File
    global Tfile
    if Tfile=='':
        for file in glob.glob(os.path.join(Script_Dir, '*test*')):
            Tfile = file

    if Tfile!='' and os.path.isfile(Tfile):
        print('Tests file found: ', Tfile)
    else:
        print('Tests file not found')
        exit()

    #Serch for Pseudo File
    global Pfile
    if Pfile == '':
        for file in os.listdir(Script_Dir):
            if file.endswith(".inp") and os.path.join(Script_Dir,file) != Tfile:
                Pfile = os.path.join(Script_Dir, file)

    if Pfile != '' and os.path.isfile(Pfile):
        print('Pseudopotential file found: ', Pfile)
    else:
        print('Pseudopotential file not found')
        exit()

    #Search for ATOM Scripts
    if os.path.isfile(pg):
        print('pg.sh found: ', pg)
    else:
        print('pg.sh not found, edit script and correct path.') 
        exit()

    if os.path.isfile(pt):
        print('pt.sh found: ', pt)
    else:
        print('pt.sh not found, edit script and correct path.') 
        exit()


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
            number_tests +=1
            test_locations.append(i)
    test_locations.append(len(lines))

    for i in range(number_tests):
        tests.append(read_ae(lines[test_locations[i]:test_locations[i+1]]))

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

def write_test(tests, file='test.inp'):
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
        print('run_type should be ae or pt, got: '+ae_config['run_type'])

    if ae_config['XC_flavor'] not in ['xc', 'wi', 'hl', 'gl', 'bh', 'ca', 'pw', 'pb', 'wp', 'rp', 'rv', 'ps', 'wc', 'jo', 'jh', 'go', 'gh', 'am', 'bl', 'vw', 'vl', 'vk', 'vc', 'vb', 'vv']:
        error = True
        print('XC_flavor should be xc, wi, hl, gl, bh, ca, pw, pb, wp, rp, rv, ps, wc, jo, jh, go, gh, am, bl, vw, vl, vk, vc, vb or vv, got: '+ae_config['XC_flavor'])

    if ae_config['XC_flavor'] == 'xc' and (not len(ae_config['XC_id']) == 8 or not ae_config['XC_id'][0] == '0' or not ae_config['XC_id'][4] == '0'):
        error = True
        print('XC_id has to be a 8 digit integer that follows 0XXX0YYY where XXX and YYY are libxc IDs. XXX is exchange and YYY is correlation. Got: '+ae_config['XC_id'])

    if error:
        raise Valueerror('Input has errors')

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
    
def write_ae(ae_config, output=[]):
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

def check_pseudo(pseudo_config = Pseudo_Config):
    error = False
    if pseudo_config['run_type'] not in ['pg', 'pe']:
        error = True
        print('run_type should be pg or pe, got: '+pseudo_config['run_type'])

    if pseudo_config['PS_flavor'] not in ['hsc', 'ker', 'tm2']:
        error = True
        print('PS_flavor should be hsc, ker or tm2, got: '+pseudo_config['PS_flavor'])

    if pseudo_config['XC_flavor'] not in ['xc', 'wi', 'hl', 'gl', 'bh', 'ca', 'pw', 'pb', 'wp', 'rp', 'rv', 'ps', 'wc', 'jo', 'jh', 'go', 'gh', 'am', 'bl', 'vw', 'vl', 'vk', 'vc', 'vb', 'vv']:
        error = True
        print('XC_flavor should be xc, wi, hl, gl, bh, ca, pw, pb, wp, rp, rv, ps, wc, jo, jh, go, gh, am, bl, vw, vl, vk, vc, vb or vv, got: '+pseudo_config['XC_flavor'])


    if pseudo_config['XC_flavor'] == 'xc' and (not len(pseudo_config['XC_id']) == 8 or not pseudo_config['XC_id'][0] == '0' or not pseudo_config['XC_id'][4] == '0'):
        error = True
        print('XC_id has to be a 8 digit integer that follows 0XXX0YYY where XXX and YYY are libxc IDs. XXX is correlation and YYY is exchange. got: '+pseudo_config['XC_id'])

    if error:
        raise Valueerror('Input has errors')

def read_pseudo(file, pseudo_config = Pseudo_Config):
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].split('#')[0]
    
    lines = [line for line in lines if not line.isspace()]
    lines = [line.rstrip() for line in lines]
    lines = [line for line in lines if line.strip()]

    # first line
    pseudo_config['run_type'] = lines[0].split()[0][:2]
    pseudo_config['title'] = lines[0][lines[0].find(pseudo_config['run_type']) + 2:]

    # second line
    pseudo_config['PS_flavor'] =  lines[1].split()[0]
    pseudo_config['logder_r'] =  float(lines[1].split()[1])

    # third line
    pseudo_config['chemical_symbol'] = lines[2].split()[0][-2:]
    if lines[2].split()[1][-1] in ['r', 's']:
        pseudo_config['spin_relativistic'] = lines[2].split()[1][-1]
        pseudo_config['XC_flavor'] = lines[2].split()[1][-3:-1]
    else:
        pseudo_config['XC_flavor'] = lines[2].split()[1][-2:]

    if pseudo_config['XC_flavor'] == 'xc':
        pseudo_config['XC_id'] = lines[2].split()[2]

    # fourth line
    pseudo_config['fractional_atomic_number'] = float(lines[3].split()[0])

    # fifth line
    pseudo_config['norbs_core'] = int(lines[4].split()[0])
    pseudo_config['norbs_valence'] = int(lines[4].split()[1])

    # orbitals block
    pseudo_config['orbital_block'] = []
    for i in range(1, pseudo_config['norbs_valence'] + 1):
        orbitals = lines[4+i].split()
        if len(orbitals) > 3:
            orbitals = [int(orbitals[0]), int(orbitals[1]), float(orbitals[2]), float(orbitals[3])]
        else:
            orbitals = [int(orbitals[0]), int(orbitals[1]), float(orbitals[2]), 0.0]
            
        pseudo_config['orbital_block'].append(orbitals)

    # last line
    i = pseudo_config['norbs_valence'] + 5
    if len(lines[i].split()) > 5:
        cutoff_radii = lines[i].split()[:6]
    elif len(lines[i].split()) > 3:
        cutoff_radii = lines[i].split()[:4]

    cutoff_radii = [float(x) for x in cutoff_radii]
    pseudo_config['rs'] = cutoff_radii[0]
    pseudo_config['rp'] = cutoff_radii[1]
    pseudo_config['rd'] = cutoff_radii[2]
    pseudo_config['rf'] = cutoff_radii[3]

    if len(cutoff_radii) > 5:
        pseudo_config['rcore'] = cutoff_radii[5]

    check_pseudo(pseudo_config)

def print_pseudo(pseudo_config = Pseudo_Config):
    # (format 3x,a2,a50)
    print(f"{'':3}{pseudo_config['run_type']:2}{'':1}{pseudo_config['title']:49}")
    # (format 8x, a3, f9.3)
    print(f"{'':8}{pseudo_config['PS_flavor']:3}{pseudo_config['logder_r']:9.3f}")
    # (format 3x,a2,3x,a2,a1,1x,i8)
    print(f"{'':3}{pseudo_config['chemical_symbol']:2}{'':3}{pseudo_config['XC_flavor']:2}{pseudo_config['spin_relativistic']:1}{'':1}{pseudo_config['XC_id']:8}")
    # (format f10.5)
    print(f"{pseudo_config['fractional_atomic_number']:10.1f}")
    # (format 2i5)
    print(f"{pseudo_config['norbs_core']:5d}{pseudo_config['norbs_valence']:5d}")
    # block of orbitals (format 2i5,2f10.3)
    for line in pseudo_config['orbital_block']:
        print(f"{line[0]:5d}{line[1]:5d}{line[2]:10.3f}{line[3]:10.3f}")
    
    # (format 6f10.5)
    print(f"{pseudo_config['rs']:10.5f}{pseudo_config['rp']:10.5f}{pseudo_config['rd']:10.5f}{pseudo_config['rf']:10.5f}{pseudo_config['rcore_flag']:10.5f}{pseudo_config['rcore']:10.5f}")
    print('')

def write_pseudo(pseudo_config=Pseudo_Config, file='temporary.inp'):
    output = []

    # (format 3x,a2,a50)
    output.append(f"{'':3}{pseudo_config['run_type']:2}{'':1}{pseudo_config['title']:49}")
    # (format 8x, a3, f9.3)
    output.append(f"{'':8}{pseudo_config['PS_flavor']:3}{pseudo_config['logder_r']:9.3f}")
    # (format 3x,a2,3x,a2,a1,1x,i8)
    output.append(f"{'':3}{pseudo_config['chemical_symbol']:2}{'':3}{pseudo_config['XC_flavor']:2}{pseudo_config['spin_relativistic']:1}{'':1}{pseudo_config['XC_id']:8}")
    # (format f10.5)
    output.append(f"{pseudo_config['fractional_atomic_number']:10.1f}")
    # (format 2i5)
    output.append(f"{pseudo_config['norbs_core']:5d}{pseudo_config['norbs_valence']:5d}")
    # block of orbitals (format 2i5,2f10.3)
    for line in pseudo_config['orbital_block']:
        output.append(f"{line[0]:5d}{line[1]:5d}{line[2]:10.3f}{line[3]:10.3f}")
    # (format 6f10.5)
    output.append(f"{pseudo_config['rs']:10.5f}{pseudo_config['rp']:10.5f}{pseudo_config['rd']:10.5f}{pseudo_config['rf']:10.5f}{pseudo_config['rcore_flag']:10.5f}{pseudo_config['rcore']:10.5f}")

    # Join all parts with newlines and write to the file
    with open(file, 'w') as f:
        f.write("\n".join(output))


if __name__ == '__main__':
    main()
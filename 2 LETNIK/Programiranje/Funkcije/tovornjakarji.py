# =============================================================================
# Tovornjakarji
#
# V Republiki Banana zaradi utrujenosti in pomanjkanja koncentracije tovornjakarji vse pogosteje povzročajo nesreče. Vlada se je odločila, da bodo zadevo rešili s sprejetjem novih zakonov o tovornem prevozu. Najprej morajo analitiki preučiti navade voznikov in v ta namen so pridobili kopico podatkov. Za vsakega tovornjakarja imajo seznam prevoženih razdalj po posameznih dnevih.
# Število 0 pomeni, da je tovornjakar tisti dan počival. Na primer seznam
# 
#     [350, 542.5, 0, 602, 452.5, 590.5, 0, 248]
# 
# pomeni, da je tovornjakar prvi dan prevozil 350 km, drugi dan
# 542.5 km, tretji dan je počival itd.
# =====================================================================@005672=
# 1. podnaloga
# Sestavite funkcijo `pocitek_in_povprecje(voznja)`, ki kot argument
# dobi zgoraj opisani seznam (prevožene razdalje po posameznih dnevih)
# in vrne par števil. Prvo število naj bo število dni, ko je kamionist
# počival. Druga število naj bo povprečna dnevna prevožena razdalja, pri
# čemer upoštevamo samo tiste dneve, ko ni počival. Primer:
# 
#     >>> pocitek_in_povprecje([200, 300, 0, 100])
#     (1, 200.0)
# 
# Kamionist je torej enkrat počival. Ob dnevih, ko ni počival, pa je
# v povprečju prevozil 200 km.
# =============================================================================
def pocitek_in_povprecje(voznja):
    pocitek=0
    povprecje=0
    for i in voznja:
        if i==0:
            pocitek+=1
        elif i != 0:
            povprecje+=i
    return (pocitek, povprecje/(len(voznja)-pocitek))
# =====================================================================@005673=
# 2. podnaloga
# Napišite funkcijo `primerjaj(prvi, drugi)`, ki primerja vožnjo dveh
# kamionistov. Funkcija kot argumenta dobi dva enako dolga seznama
# `prvi` in `drugi`, ki opisujeta vožnjo dveh kamionistov, ki sta se
# podala na isto pot. Funkcije naj sestavi in vrne nov seznam, v katerem
# je za vsak dan zapisano, kdo je do tega dne prevozil večjo razdaljo:
# `1`, če je bil to 1. kamionist; `2`, če je bil to 2. kamionist in
# `'enako'`, če sta oba prevozila enako razdaljo. Primer:
# 
#     >>> primerjaj([200, 300, 100], [200, 200, 300])
#     ['enako', 1, 2]
#     >>> primerjaj([500, 100, 100], [100, 200, 200])
#     [1, 1, 1]
# =============================================================================
def primerjaj(prvi, drugi):
    primerjava=[]
    vsota_1=0
    vsota_2=0
    for i in range(len(prvi)):
        vsota_1+=prvi[i]
        vsota_2+=drugi[i]
        if vsota_1 < vsota_2:
            primerjava.append(2)
        elif  vsota_1 > vsota_2:
            primerjava.append(1)
        else: 
            primerjava.append("enako")
    return primerjava
# =====================================================================@005674=
# 3. podnaloga
# Vlada je uzakonila naslednja pravila:
# 
# * Povprečna prevožena razdalja v treh zaporednih dneh ne sme biti več
#   kot 500 km. (Štejemo tudi dneve, ko je kamionist počival.)
# * Kamionist ne sme brez počitka voziti več kot 5 dni v zaporedju.
# 
# Sestavite funkcijo `po_pravilih(voznja)`, ki vrne `True`, če se je
# kamionist držal predpisov, in `False`, če se jih ni.
# 
#     >>> po_pravilih([50, 200, 300, 20, 100, 60])
#     False
#     >>> po_pravilih([600, 200, 0, 300, 300, 0, 600, 600, 400])
#     False
#     >>> po_pravilih([600, 600, 0, 600, 600])
#     True
# =============================================================================
def po_pravilih(voznja):
    spanje=0
    povprecje=0
    for d in voznja: 
        if d==0:
            spanje=0
        else:
            spanje+=1
        if spanje > 5:
            return False
        
    for i in range(len(voznja)-2):
        povprecje= (voznja[i] + voznja[i+1] + voznja[i+2]) /3 
        if povprecje > 500:
            return False
    return True
# =====================================================================@005675=
# 4. podnaloga
# Podatke za več kamionistov so analitiki združili v en sam seznam.
# Zgled takega seznama:
# 
#     vsi_skupaj = [
#         [50, 200, 300, 20, 100, 60],
#         [600, 600, 0, 600, 600, 0, 600, 600],
#         [600, 200, 0, 300, 300, 0, 600, 600, 400]
#     ]
# 
# Sestavite funkcijo `preveri_vse(seznam_vozenj)`, ki dobi kot argument
# opisani seznam in vrne seznam logičnih vrednosti. Za vsakega kamionista
# naj funkcija ugotovi, če je vozil po predpisih. Primer:
# 
#     >>> preveri_vse(vsi_skupaj)
#     [False, True, False]
# =============================================================================

def preveri_vse(seznam_vozenj):
    rezultati=[]
    for k in seznam_vozenj:
        rezultati.append(po_pravilih(k))
    return rezultati



































































































# ============================================================================@

'Če vam Python sporoča, da je v tej vrstici sintaktična napaka,'
'se napaka v resnici skriva v zadnjih vrsticah vaše kode.'

'Kode od tu naprej NE SPREMINJAJTE!'


















































import json, os, re, sys, shutil, traceback, urllib.error, urllib.request


import io, sys
from contextlib import contextmanager

class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end='')
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end='')
        return line

class Check:
    @staticmethod
    def has_solution(part):
        return part['solution'].strip() != ''

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part['valid'] = True
            part['feedback'] = []
            part['secret'] = []
        Check.current_part = None
        Check.part_counter = None

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part['feedback'].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part['valid'] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed))
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted([(Check.clean(k, digits, typed), Check.clean(v, digits, typed)) for (k, v) in x.items()])
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get('clean', clean)
        Check.current_part['secret'].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error('Izraz {0} vrne {1!r} namesto {2!r}.',
                        expression, actual_result, expected_result)
            return False
        else:
            return True

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        exec(code, global_env)
        errors = []
        for (x, v) in expected_state.items():
            if x not in global_env:
                errors.append('morajo nastaviti spremenljivko {0}, vendar je ne'.format(x))
            elif clean(global_env[x]) != clean(v):
                errors.append('nastavijo {0} na {1!r} namesto na {2!r}'.format(x, global_env[x], v))
        if errors:
            Check.error('Ukazi\n{0}\n{1}.', statements,  ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, 'w', encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part['feedback'][:]
        yield
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n    '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}', filename, '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part['feedback'][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get('stringio')('\n'.join(content) + '\n')
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n  '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}', '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error('Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}', filename, (line_width - 7) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(expression, global_env)
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal:
            return True
        else:
            Check.error('Program izpiše{0}  namesto:\n  {1}', (line_width - 13) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ['\n']
        else:
            expected_lines += (actual_len - expected_len) * ['\n']
        equal = True
        line_width = max(len(actual_line.rstrip()) for actual_line in actual_lines + ['Program izpiše'])
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append('{0} {1} {2}'.format(out.ljust(line_width), '|' if out == given else '*', given))
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get('update_env', update_env):
            global_env = dict(global_env)
        global_env.update(Check.get('env', env))
        return global_env

    @staticmethod
    def generator(expression, expected_values, should_stop=None, further_iter=None, clean=None, env=None, update_env=None):
        from types import GeneratorType
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error("Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                                iteration, expression, actual_value, expected_value)
                    return False
            for _ in range(Check.get('further_iter', further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False
        
        if Check.get('should_stop', should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print('{0}. podnaloga je brez rešitve.'.format(i + 1))
            elif not part['valid']:
                print('{0}. podnaloga nima veljavne rešitve.'.format(i + 1))
            else:
                print('{0}. podnaloga ima veljavno rešitev.'.format(i + 1))
            for message in part['feedback']:
                print('  - {0}'.format('\n    '.join(message.splitlines())))

    settings_stack = [{
        'clean': clean.__func__,
        'encoding': None,
        'env': {},
        'further_iter': 0,
        'should_stop': False,
        'stringio': VisibleStringIO,
        'update_env': False,
    }]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs))
                             if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get('env'))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get('stringio'):
            yield
        else:
            with Check.set(stringio=stringio):
                yield


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding='utf-8') as f:
            source = f.read()
        part_regex = re.compile(
            r'# =+@(?P<part>\d+)=\s*\n' # beginning of header
            r'(\s*#( [^\n]*)?\n)+?'     # description
            r'\s*# =+\s*?\n'            # end of header
            r'(?P<solution>.*?)'        # solution
            r'(?=\n\s*# =+@)',          # beginning of next part
            flags=re.DOTALL | re.MULTILINE
        )
        parts = [{
            'part': int(match.group('part')),
            'solution': match.group('solution')
        } for match in part_regex.finditer(source)]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]['solution'] = parts[-1]['solution'].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = '{0}.{1}'.format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    'part': part['part'],
                    'solution': part['solution'],
                    'valid': part['valid'],
                    'secret': [x for (x, _) in part['secret']],
                    'feedback': json.dumps(part['feedback']),
                }
                if 'token' in part:
                    submitted_part['token'] = part['token']
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode('utf-8')
        headers = {
            'Authorization': token,
            'content-type': 'application/json'
        }
        request = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response['attempts']:
            part['feedback'] = json.loads(part['feedback'])
            updates[part['part']] = part
        for part in old_parts:
            valid_before = part['valid']
            part.update(updates.get(part['part'], {}))
            valid_after = part['valid']
            if valid_before and not valid_after:
                wrong_index = response['wrong_indices'].get(str(part['part']))
                if wrong_index is not None:
                    hint = part['secret'][wrong_index][1]
                    if hint:
                        part['feedback'].append('Namig: {}'.format(hint))


    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NjcyfQ:1iIyci:uka_CO7k_-uphk782FGiC5QBfqA'
        try:
            test_data = [
                ("""pocitek_in_povprecje([200, 300, 0, 100])""", (1, 200)),
                ("""pocitek_in_povprecje([350, 542.5, 0, 602, 452.5, 590.5, 0, 248])""", (2, 464.25)),
                ("""pocitek_in_povprecje([0, 0, 350, 542.5, 0, 602, 452.5, 590.5, 0, 248, 0, 0])""", (6, 464.25)),
                ("""pocitek_in_povprecje([100, 200, 100, 200, 100, 200, 100, 200])""", (0, 150)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NjczfQ:1iIyci:wad_E57ZWkaDtSL9y47cSf7JQMY'
        try:
            test_data = [
                ("""primerjaj([200, 300, 100], [200, 200, 300])""",
                 ['enako', 1, 2]),
                ("""primerjaj([500, 100, 100], [100, 200, 200])""",
                 [1, 1, 1]),
                ("""primerjaj([500, 0, 300, 100, 300, 100, 300], [500, 0, 200, 300, 100, 300, 100])""",
                 ['enako', 'enako', 1, 2, 1, 2, 1]),
                ("""primerjaj([350, 542.5, 0, 602, 452.5, 590.5, 0, 248], [350, 0, 0, 602, 542.5, 452.5, 590.5, 248])""",
                 ['enako', 1, 1, 1, 1, 1, 'enako', 'enako']),
                ("""primerjaj([100, 200, 100, 200, 100, 200, 100, 200], [200, 100, 200, 100, 200, 100, 200, 100])""",
                 [2, 'enako', 2, 'enako', 2, 'enako', 2, 'enako']),
                ("""primerjaj([300, 100, 300, 100, 300], [200, 300, 100, 300, 100])""",
                 [1, 2, 1, 2, 1]),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1Njc0fQ:1iIyci:A6FTcAl4mrFJ9q0-wKktprjGkY8'
        try:
            test_data = [
                ("""po_pravilih([200, 300, 0, 100])""", True),
                ("""po_pravilih([500, 500, 500, 500, 500, 0, 500, 500, 500, 500, 500, 0, 500, 500, 500, 500, 500])""", True),
                ("""po_pravilih([50, 200, 300, 20, 100, 60])""", False),
                ("""po_pravilih([600, 200, 0, 300, 300, 0, 600, 600, 400])""", False),
                ("""po_pravilih([600, 600, 0, 600, 600, 0, 600, 600])""", True),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1Njc1fQ:1iIyci:olI4a8zMZsooB_H-y6GRvM0s55c'
        try:
            test_data = [
                ("""preveri_vse([[50, 200, 300, 20, 100, 60], [600, 600, 0, 600, 600, 0, 600, 600], [600, 200, 0, 300, 300, 0, 600, 600, 400]])""", [False, True, False]),
                ("""preveri_vse([[200, 300, 0, 100], [500, 500, 500, 500, 500, 0, 500, 500, 500, 500, 500, 0, 500, 500, 500, 500, 500], [50, 200, 300, 20, 100, 60], [600, 200, 0, 300, 300, 0, 600, 600, 400], [600, 600, 0, 600, 600, 0, 600, 600]])""", [True, True, False, False, True]),
                ("""preveri_vse([[50, 200, 300, 20, 100, 60], [600, 200, 0, 300, 300, 0, 600, 600, 400], [200, 300, 0, 100], [500, 500, 500, 500, 500, 0, 500, 500, 500, 500, 500, 0, 500, 500, 500, 500, 500], [600, 600, 0, 600, 600, 0, 600, 600]])""", [False, False, True, True, True]),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    print('Shranjujem rešitve na strežnik... ', end="")
    try:
        url = 'https://www.projekt-tomo.si/api/attempts/submit/'
        token = 'Token b8c7df44b9c20c8359c39be75a6d39215c0b0a7f'
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        print('PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE! Poskusite znova.')
    else:
        print('Rešitve so shranjene.')
        update_attempts(Check.parts, response)
        if 'update' in response:
            print('Posodabljam datoteko... ', end="")
            backup_filename = backup(filename)
            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(response['update'])
            print('Stara datoteka je bila preimenovana v {0}.'.format(backup_filename))
            print('Če se datoteka v urejevalniku ni osvežila, jo zaprite ter ponovno odprite.')
    Check.summarize()

if __name__ == '__main__':
    _validate_current_file()

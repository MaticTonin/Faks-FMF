# =============================================================================
# Rimski imperij vrača udarec
#
# Pri tej nalogi boste napisali funkcijo, ki bo za dano število vrnila
# niz z ustrezno rimsko številko. Nato boste napisali še funkcijo, ki bo
# rimsko številko spremenila nazaj v število.
# 
# Preden začnete si na oglejte članek o rimskih številkah na Wikipediji:
# [Rimske številke](http://sl.wikipedia.org/wiki/Rimske_%C5%A1tevilke).
# Lahko si preberete tudi opis na portalu ProjectEuler.net:
# [About… Roman Numerals](https://projecteuler.net/about=roman_numerals).
# =====================================================================@005738=
# 1. podnaloga
# Napišite funkcijo `v_rimsko(stevilo)`, ki kot argument dobi celo
# število med 1 in 3999 (vključno z 1 in 3999). Funkcija naj sestavi
# in vrne niz, ki vsebuje _standardno_ rimsko številko, ki predstavlja
# število `stevilo`. Zgled:
# 
#     >>> v_rimsko(2013)
#     'MMXIII'
# =============================================================================
tabela_rimskih = [
  ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'],
  ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC'],
  ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'],
  ['', 'M', 'MM', 'MMM']
]


    
def v_rimsko(stevilo):
    rimska=""
  
    for i in range(4):
        rimska=tabela_rimskih[i][stevilo % 10]+ rimska
        stevilo //= 10
    return rimska
# =====================================================================@005739=
# 2. podnaloga
# Napišite še funkcijo `v_arabsko(niz)`, ki bo dobila niz `niz`, ki
# predstavlja rimsko številko. Funkcija naj naredi ravno obratno kot
# funkcija `v_rimsko`, tj. vrne naj ustrezno število, ki ga predstavlja
# dana rimska številka. Če `niz` ne predstavlja veljavnege rimske
# številke, naj funkcija vrne `None`. Zgled:
# 
#     >>> v_arabsko('MMXIII')
#     2013
# 
# Nasvet: Najprej sestavi slovar (ta naj bo definiran že pred funkcijo
# `v_arabsko`), ki kot ključe vsebuje rimske številke, kot vrednosti
# pa ustrezna števila. Funkcija `v_arabsko` naj le uporabi ta slovar.
# =============================================================================
rimski_slovar={v_rimsko(i): i for i in range (1,4000)}
def v_arabsko(niz):
    if niz not in rimski_slovar:
        return None
    else:
        return rimski_slovar[niz]
   




































































































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
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NzM4fQ:1iMDba:-XkQ9fX6jDiYI8T7Lbh7ypKXLFM'
        try:
            testni_primeri = [(2013, 'MMXIII'), (1, 'I'), (2, 'II'), (3, 'III'), (4, 'IV'), (5, 'V'), (6, 'VI'), (7, 'VII'),
            (8, 'VIII'), (9, 'IX'), (10, 'X'), (11, 'XI'), (12, 'XII'), (13, 'XIII'), (14, 'XIV'), (15, 'XV'),
            (16, 'XVI'), (17, 'XVII'), (18, 'XVIII'), (19, 'XIX'), (20, 'XX'), (21, 'XXI'), (22, 'XXII'),
            (23, 'XXIII'), (24, 'XXIV'), (25, 'XXV'), (26, 'XXVI'), (27, 'XXVII'), (28, 'XXVIII'), (29, 'XXIX'),
            (30, 'XXX'), (31, 'XXXI'), (32, 'XXXII'), (33, 'XXXIII'), (34, 'XXXIV'), (35, 'XXXV'), (36, 'XXXVI'),
            (37, 'XXXVII'), (38, 'XXXVIII'), (39, 'XXXIX'), (40, 'XL'), (41, 'XLI'), (42, 'XLII'), (43, 'XLIII'),
            (44, 'XLIV'), (45, 'XLV'), (46, 'XLVI'), (47, 'XLVII'), (48, 'XLVIII'), (49, 'XLIX'), (50, 'L'),
            (51, 'LI'), (52, 'LII'), (53, 'LIII'), (54, 'LIV'), (55, 'LV'), (56, 'LVI'), (57, 'LVII'),
            (58, 'LVIII'), (59, 'LIX'), (60, 'LX'), (61, 'LXI'), (62, 'LXII'), (63, 'LXIII'), (64, 'LXIV'),
            (65, 'LXV'), (66, 'LXVI'), (67, 'LXVII'), (68, 'LXVIII'), (69, 'LXIX'), (70, 'LXX'), (71, 'LXXI'),
            (72, 'LXXII'), (73, 'LXXIII'), (74, 'LXXIV'), (75, 'LXXV'), (76, 'LXXVI'), (77, 'LXXVII'),
            (78, 'LXXVIII'), (79, 'LXXIX'), (80, 'LXXX'), (81, 'LXXXI'), (82, 'LXXXII'), (83, 'LXXXIII'),
            (84, 'LXXXIV'), (85, 'LXXXV'), (86, 'LXXXVI'), (87, 'LXXXVII'), (88, 'LXXXVIII'), (89, 'LXXXIX'),
            (90, 'XC'), (91, 'XCI'), (92, 'XCII'), (93, 'XCIII'), (94, 'XCIV'), (95, 'XCV'), (96, 'XCVI'),
            (97, 'XCVII'), (98, 'XCVIII'), (99, 'XCIX'), (100, 'C'), (908, 'CMVIII'), (451, 'CDLI'), (853, 'DCCCLIII'),
            (384, 'CCCLXXXIV'), (572, 'DLXXII'), (2787, 'MMDCCLXXXVII'), (1930, 'MCMXXX'), (3843, 'MMMDCCCXLIII'),
            (2156, 'MMCLVI'), (3237, 'MMMCCXXXVII'), (1208, 'MCCVIII'), (3162, 'MMMCLXII'), (3734, 'MMMDCCXXXIV'),
            (2958, 'MMCMLVIII'), (1349, 'MCCCXLIX'), (2527, 'MMDXXVII'), (2922, 'MMCMXXII'), (3928, 'MMMCMXXVIII'), 
            (1388, 'MCCCLXXXVIII'), (2725, 'MMDCCXXV'), (766, 'DCCLXVI'), (3204, 'MMMCCIV'), (1585, 'MDLXXXV'),
            (798, 'DCCXCVIII'), (1299, 'MCCXCIX'), (2709, 'MMDCCIX'), (2382, 'MMCCCLXXXII'), (3104, 'MMMCIV'),
            (1527, 'MDXXVII'), (3184, 'MMMCLXXXIV'), (301, 'CCCI'), (848, 'DCCCXLVIII'), (133, 'CXXXIII'), 
            (3145, 'MMMCXLV'), (1217, 'MCCXVII'), (428, 'CDXXVIII'), (745, 'DCCXLV'), (2753, 'MMDCCLIII'),
            (2125, 'MMCXXV'), (2063, 'MMLXIII'), (559, 'DLIX'), (240, 'CCXL'), (3411, 'MMMCDXI'), (448, 'CDXLVIII'),
            (2941, 'MMCMXLI'), (1715, 'MDCCXV'), (1309, 'MCCCIX'), (3449, 'MMMCDXLIX'), (952, 'CMLII'),
            (3285, 'MMMCCLXXXV'), (2691, 'MMDCXCI'), (3158, 'MMMCLVIII'), (3143, 'MMMCXLIII'), (2503, 'MMDIII'), 
            (546, 'DXLVI'), (2912, 'MMCMXII'), (2173, 'MMCLXXIII'), (2026, 'MMXXVI'), (1756, 'MDCCLVI'),
            (801, 'DCCCI'), (1862, 'MDCCCLXII'), (823, 'DCCCXXIII'), (670, 'DCLXX'), (2521, 'MMDXXI'), (892, 'DCCCXCII'), 
            (2836, 'MMDCCCXXXVI'), (3027, 'MMMXXVII'), (2682, 'MMDCLXXXII'), (3901, 'MMMCMI'), (3430, 'MMMCDXXX'), 
            (2924, 'MMCMXXIV'), (1253, 'MCCLIII'), (3403, 'MMMCDIII'), (3238, 'MMMCCXXXVIII'), (3112, 'MMMCXII'),
            (3327, 'MMMCCCXXVII'), (1933, 'MCMXXXIII'), (1340, 'MCCCXL'), (2596, 'MMDXCVI'), (3394, 'MMMCCCXCIV'), 
            (1992, 'MCMXCII'), (2838, 'MMDCCCXXXVIII'), (110, 'CX'), (3603, 'MMMDCIII'), (235, 'CCXXXV'), (2530, 'MMDXXX'),
            (2261, 'MMCCLXI'), (3625, 'MMMDCXXV'), (2630, 'MMDCXXX'), (2667, 'MMDCLXVII'), (1223, 'MCCXXIII'), 
            (1291, 'MCCXCI'), (733, 'DCCXXXIII'), (851, 'DCCCLI'), (2647, 'MMDCXLVII'), (3847, 'MMMDCCCXLVII'), 
            (2977, 'MMCMLXXVII'), (1690, 'MDCXC'), (1925, 'MCMXXV'), (3602, 'MMMDCII')]
            
            for arabsko, rimsko in testni_primeri:
                if not Check.equal("v_rimsko({0})".format(arabsko), rimsko):
                    break  # Test has failed!
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NzM5fQ:1iMDba:BnEQ5IAJNuXOtDQGQ61H4mwTk2s'
        try:
            testni_primeri = [(1, 'I'), (2, 'II'), (3, 'III'), (4, 'IV'), (5, 'V'), (6, 'VI'), (7, 'VII'),
            (8, 'VIII'), (9, 'IX'), (10, 'X'), (11, 'XI'), (12, 'XII'), (13, 'XIII'), (14, 'XIV'), (15, 'XV'),
            (16, 'XVI'), (17, 'XVII'), (18, 'XVIII'), (19, 'XIX'), (20, 'XX'), (21, 'XXI'), (22, 'XXII'),
            (23, 'XXIII'), (24, 'XXIV'), (25, 'XXV'), (26, 'XXVI'), (27, 'XXVII'), (28, 'XXVIII'), (29, 'XXIX'),
            (30, 'XXX'), (31, 'XXXI'), (32, 'XXXII'), (33, 'XXXIII'), (34, 'XXXIV'), (35, 'XXXV'), (36, 'XXXVI'),
            (37, 'XXXVII'), (38, 'XXXVIII'), (39, 'XXXIX'), (40, 'XL'), (41, 'XLI'), (42, 'XLII'), (43, 'XLIII'),
            (44, 'XLIV'), (45, 'XLV'), (46, 'XLVI'), (47, 'XLVII'), (48, 'XLVIII'), (49, 'XLIX'), (50, 'L'),
            (51, 'LI'), (52, 'LII'), (53, 'LIII'), (54, 'LIV'), (55, 'LV'), (56, 'LVI'), (57, 'LVII'),
            (58, 'LVIII'), (59, 'LIX'), (60, 'LX'), (61, 'LXI'), (62, 'LXII'), (63, 'LXIII'), (64, 'LXIV'),
            (65, 'LXV'), (66, 'LXVI'), (67, 'LXVII'), (68, 'LXVIII'), (69, 'LXIX'), (70, 'LXX'), (71, 'LXXI'),
            (72, 'LXXII'), (73, 'LXXIII'), (74, 'LXXIV'), (75, 'LXXV'), (76, 'LXXVI'), (77, 'LXXVII'),
            (78, 'LXXVIII'), (79, 'LXXIX'), (80, 'LXXX'), (81, 'LXXXI'), (82, 'LXXXII'), (83, 'LXXXIII'),
            (84, 'LXXXIV'), (85, 'LXXXV'), (86, 'LXXXVI'), (87, 'LXXXVII'), (88, 'LXXXVIII'), (89, 'LXXXIX'),
            (90, 'XC'), (91, 'XCI'), (92, 'XCII'), (93, 'XCIII'), (94, 'XCIV'), (95, 'XCV'), (96, 'XCVI'),
            (97, 'XCVII'), (98, 'XCVIII'), (99, 'XCIX'), (100, 'C'), (908, 'CMVIII'), (451, 'CDLI'), (853, 'DCCCLIII'),
            (384, 'CCCLXXXIV'), (572, 'DLXXII'), (2787, 'MMDCCLXXXVII'), (1930, 'MCMXXX'), (3843, 'MMMDCCCXLIII'),
            (2156, 'MMCLVI'), (3237, 'MMMCCXXXVII'), (1208, 'MCCVIII'), (3162, 'MMMCLXII'), (3734, 'MMMDCCXXXIV'),
            (2958, 'MMCMLVIII'), (1349, 'MCCCXLIX'), (2527, 'MMDXXVII'), (2922, 'MMCMXXII'), (3928, 'MMMCMXXVIII'), 
            (1388, 'MCCCLXXXVIII'), (2725, 'MMDCCXXV'), (766, 'DCCLXVI'), (3204, 'MMMCCIV'), (1585, 'MDLXXXV'),
            (798, 'DCCXCVIII'), (1299, 'MCCXCIX'), (2709, 'MMDCCIX'), (2382, 'MMCCCLXXXII'), (3104, 'MMMCIV'),
            (1527, 'MDXXVII'), (3184, 'MMMCLXXXIV'), (301, 'CCCI'), (848, 'DCCCXLVIII'), (133, 'CXXXIII'), 
            (3145, 'MMMCXLV'), (1217, 'MCCXVII'), (428, 'CDXXVIII'), (745, 'DCCXLV'), (2753, 'MMDCCLIII'),
            (2125, 'MMCXXV'), (2063, 'MMLXIII'), (559, 'DLIX'), (240, 'CCXL'), (3411, 'MMMCDXI'), (448, 'CDXLVIII'),
            (2941, 'MMCMXLI'), (1715, 'MDCCXV'), (1309, 'MCCCIX'), (3449, 'MMMCDXLIX'), (952, 'CMLII'),
            (3285, 'MMMCCLXXXV'), (2691, 'MMDCXCI'), (3158, 'MMMCLVIII'), (3143, 'MMMCXLIII'), (2503, 'MMDIII'), 
            (546, 'DXLVI'), (2912, 'MMCMXII'), (2173, 'MMCLXXIII'), (2026, 'MMXXVI'), (1756, 'MDCCLVI'),
            (801, 'DCCCI'), (1862, 'MDCCCLXII'), (823, 'DCCCXXIII'), (670, 'DCLXX'), (2521, 'MMDXXI'), (892, 'DCCCXCII'), 
            (2836, 'MMDCCCXXXVI'), (3027, 'MMMXXVII'), (2682, 'MMDCLXXXII'), (3901, 'MMMCMI'), (3430, 'MMMCDXXX'), 
            (2924, 'MMCMXXIV'), (1253, 'MCCLIII'), (3403, 'MMMCDIII'), (3238, 'MMMCCXXXVIII'), (3112, 'MMMCXII'),
            (3327, 'MMMCCCXXVII'), (1933, 'MCMXXXIII'), (1340, 'MCCCXL'), (2596, 'MMDXCVI'), (3394, 'MMMCCCXCIV'), 
            (1992, 'MCMXCII'), (2838, 'MMDCCCXXXVIII'), (110, 'CX'), (3603, 'MMMDCIII'), (235, 'CCXXXV'), (2530, 'MMDXXX'),
            (2261, 'MMCCLXI'), (3625, 'MMMDCXXV'), (2630, 'MMDCXXX'), (2667, 'MMDCLXVII'), (1223, 'MCCXXIII'), 
            (1291, 'MCCXCI'), (733, 'DCCXXXIII'), (851, 'DCCCLI'), (2647, 'MMDCXLVII'), (3847, 'MMMDCCCXLVII'), 
            (2977, 'MMCMLXXVII'), (1690, 'MDCXC'), (1925, 'MCMXXV'), (3602, 'MMMDCII'),
            (None, 'IIII'), (None, 'IVXCDCXVI')]
            for arabsko, rimsko in testni_primeri:
                if not Check.equal("v_arabsko('{0}')".format(rimsko), arabsko):
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

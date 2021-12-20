# =============================================================================
# Gensko spremenjena hrana
#
# Upravljalec restavracije te prosi za pomoč. Ljudje so ozaveščeni in
# skrbi jih gensko spremenjena hrana. Moram mu pomagati, da bo izračunal
# količino GSOja, ki bo zapisana v novem jedilniku.
# =====================================================================@005743=
# 1. podnaloga
# Kuhar je že sestavil slovar vseh jedi na jedilniku skupaj z njihovimi
# sestavinami. Sestavine so tudi podane v obliki slovarja z vnosi
# oblike `sestavina: količina`. Enota pri količini ni podadana (lahko je to
# teža, volumen, število kosov, …). Ker se je zelo mudilo, je pri jedeh,
# kjer je cela jed ena sama sestavina, pisal le prazno množico. Sestavite
# funkcijo `odpravi_okrajsave(recepti)`, ki sestavi in vrne nov slovar,
# ki ne vsebuje okrajšav. Zgled:
# 
#     >>> recepti = {'pica': {'moka': 80, 'sir': 30}, 'solata': dict()}
#     >>> odpravi_okrajsave(recepti)
#     {'solata': {'solata': 1}, 'pica': {'sir': 30, 'moka': 80}}
# =============================================================================
def odpravi_okrajsave(recepti):
    recept=dict()
    for key, value in recepti.items():
        if len(value)==0:
            recept[key]={key:1}
        else:
            recept[key]=value
    return recept
# =====================================================================@005744=
# 2. podnaloga
# V tistih množicah, ki predstavljajo recepte, je kuhar količine sestavin
# podal v gramih. Da bomo lažje računali GSO, jih je treba "normalizirat",
# da bo vse v procentih. Napišite funkcijo `normaliziraj_kolicine(recepti)`,
# ki kot argument dobi slovar receptov in vrne "normaliziran" slovar receptov
# (tj. vsota vseh količin v vsakem receptu naj bo točno 100). Zgled:
# 
#     >>> recepti = {'pica': {'moka': 80, 'sir': 30} 'solata': dict()}
#     >>> normaliziraj_kolicine(recepti)
#     {'solata': {'solata': 100.0},
#      'pica': {'sir': 27.272727272727273, 'moka': 72.72727272727273}}
# =============================================================================
def normaliziraj_kolicine(recepti):
    brez_okrajsav=odpravi_okrajsave(recepti)
    for jed, kolicina in brez_okrajsav.items():
        vsota=sum(kolicina.values())
        brez_okrajsav[jed]={key: value*100/vsota for key, value in kolicina.items()}
    return brez_okrajsav
# =====================================================================@005745=
# 3. podnaloga
# Upravljalec restavracije vam je priskrbel še seznam vseh elementarnih
# sestavin skupaj z njihovim deležem GSO. Sestavite funkcijo
# `delez_gso(recepti, elementarne)`, ki dobi slovar z recepti in slovar
# z deleži GSO v elementarnih sestavinah in poračuna deleže GSO v jedeh.
# V slovarju elementarnih sestavin so vsaj vse elementarne sestavine, ki
# se nahajajo v receptih, lahko pa jih je tudi več. Zgled:
# 
#     >>> recepti = {'pica': {'moka': 80, 'sir': 30}, 'solata': dict()}
#     >>> elementarne = {'moka': 70, 'sir': 10, 'solata': 0}
#     >>> delez_gso(recepti, elementarne)
#     {'solata': 0.0, 'pica': 53.63636363636364}
# =============================================================================
def delez_gso(recepti, elementarne):
    gso=dict()
    normaliziran=normaliziraj_kolicine(recepti)
    for jed, sestavine in normaliziran.items():
        gso_jedi=sum(value*elementarne[key] for key, value in sestavine.items()) / 100
        gso[jed]=gso_jedi
    return gso



# =====================================================================@005746=
# 4. podnaloga
# Poleg običajnega imajo v restavraciji tudi specialni jedilnik, ki vsebuje
# eksotične jedi. Te vsebujejo tudi sestavine, ki prihajajo iz "sumljivih"
# dežel in deleža GSO ni mogoče ugotoviti. Napišite še funkcijo
# `delez_gso_special(recepti, elementarne)`, ki je podobna kot v prejšni
# nalogi, le da vsaki jedi priredi "interval" (tj. par `(min_gso, max_gso)`).
# V minimalnem primeru predpostavimo povsod 0 % GSO, v maximalnem primeru
# pa povsod 100 % GSO. Zgled:
# 
#     >>> recepti = {'pica_special': {'moka': 80, 'sir': 30, 'namaz': 10}, 'solata_special': dict()}
#     >>> elementarne = {'moka': 70, 'sir': 10, 'solata': 5}
#     >>> delez_gso_special(recepti, elementarne)
#     {'pica_special': (49.16666666666667, 57.5), 'solata_special': (0.0, 100.0)}
# =============================================================================





































































































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
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NzQzfQ:1iMDbf:0Qldyohzj9totUO2ilhnK-5tIpI'
        try:
            test_data = [
                ("odpravi_okrajsave({'pica': {'moka': 80, 'sir': 30}, 'solata': dict()})",
                 {'solata': {'solata': 1}, 'pica': {'sir': 30, 'moka': 80}}),
                ("odpravi_okrajsave(dict())", dict()),
                ("odpravi_okrajsave({'voda': {}, 'toast': {'kruh': 50, 'sir': 40, 'šunka':30}})",
                 {'toast': {'kruh': 50, 'sir': 40, 'šunka': 30}, 'voda': {'voda': 1}}),
                ("odpravi_okrajsave({'voda': dict(), 'hruška': dict()})",
                 {'voda': {'voda': 1}, 'hruška': {'hruška': 1}}),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NzQ0fQ:1iMDbf:balAYSu4aRjieNJY0K2wdZhfDPo'
        try:
            test_data = [
                ("normaliziraj_kolicine({'pica': {'moka': 80, 'sir': 30}, 'solata': dict()})",
                 {'solata': {'solata': 100.0}, 'pica': {'sir': 27.272727272727273, 'moka': 72.72727272727273}}),
                ("normaliziraj_kolicine(dict())",
                 dict()),
                ("normaliziraj_kolicine({'voda': dict(), 'solata': dict()})",
                 {'solata': {'solata': 100.0}, 'voda': {'voda': 100.0}}),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NzQ1fQ:1iMDbf:lvICzCOGqwhxaL2QTq2AsenqPck'
        try:
            test_data = [
                ("delez_gso({'pica': {'moka': 80, 'sir': 30}, 'solata': dict()}, {'moka': 70, 'sir': 10, 'solata': 0})",
                 {'solata': 0.0, 'pica': 53.63636363636364}),
                ("delez_gso({'hruška': dict(), 'solata': {'olje':10, 'ledenka':80}}, {'hruška': 50, 'ledenka': 10, 'olje': 33})",
                 {'solata': 12.555555555555557, 'hruška': 50.0}),
                ("delez_gso(dict(), dict())",
                 dict()),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo0NTQ2LCJwYXJ0Ijo1NzQ2fQ:1iMDbf:pKDUaa0VQbFhsuAjGWT6TYn2VCo'
        try:
            test_data = [
                ("delez_gso_special({'pica_special': {'moka': 80, 'sir': 30, 'namaz': 10}, 'solata_special': dict()}, {'moka': 70, 'sir': 10, 'solata': 5})",
                 {'pica_special': (49.16666666666667, 57.5), 'solata_special': (0.0, 100.0)}),
                ("delez_gso_special({'pica_special': {'moka': 80, 'sir': 30, 'namaz': 10}, 'solata': dict()}, {'moka': 70, 'sir': 10, 'solata': 5})",
                 {'solata': (5.0, 5.0), 'pica_special': (49.16666666666667, 57.5)}),
                ("delez_gso_special(dict(), dict())",
                 dict()),
                ("delez_gso_special({'hruška': dict(), 'solata': {'olje': 10, 'ledenka': 80}}, {'hruška': 50, 'ledenka': 10, 'olje': 33})",
                 {'solata': (12.555555555555557, 12.555555555555557), 'hruška': (50.0, 50.0)}),
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

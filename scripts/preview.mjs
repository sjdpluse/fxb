// Internal UI verification only. The normal workstation is `python -m fxb serve`.
import {existsSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {resolve, delimiter} from 'node:path';
const local = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
const python = existsSync(local) ? resolve(local) : process.env.CODEX_PRIMARY_RUNTIME_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
// The preview container can share project packages while using its own Python 3.12 binary.
const packages = resolve('.venv/lib/python3.12/site-packages');
const pythonPath = [resolve('src'), ...(existsSync(packages) ? [packages] : [])].join(delimiter);
const child = spawn(python, ['-m', 'fxb.ui', '--preview', ...process.argv.slice(2)], {
  stdio:'inherit', env:{...process.env, PYTHONPATH:pythonPath}
});
child.on('exit', code=>process.exit(code??1));
child.on('error', error=>{process.stderr.write(error.message+'\n');process.exit(1);});
process.on('SIGTERM',()=>child.kill('SIGTERM'));
process.on('SIGINT',()=>child.kill('SIGINT'));

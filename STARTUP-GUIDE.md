# Quick Startup Guide

## Prerequisites
- Ollama installed and models pulled: `ollama pull qwen3:4b` and `ollama pull qwen3.6:35b-a3b`
- Python 3.10+ with the project installed

## First Time Setup (One-time)

1. **Set environment variables for parallel processing:**
   ```cmd
   setx OLLAMA_NUM_PARALLEL 8
   setx OLLAMA_NUM_THREADS 12
   ```

2. **Kill any existing Ollama processes:**
   ```cmd
   taskkill /IM "ollama app.exe" /F
   taskkill /IM ollama.exe /F
   ```

3. **Start Ollama** (Windows PowerShell):
   ```powershell
   Start-Process -FilePath "$env:USERPROFILE\AppData\Local\Programs\Ollama\ollama app.exe" -WindowStyle Hidden
   ```
   Or use the Ollama desktop app.

## Running the Wiki Generation Pipeline

### In a Command Prompt:

#### 1st cmd window

```cmd
set OLLAMA_VULKAN=1 && set OLLAMA_NUM_PARALLEL=6 && setx OLLAMA_NUM_THREADS 6 && ollama serve
```

- Using VULKAN used integrated grpahics no CPU, otherwise only CPU

if it already runs do:

```cmd
taskkill /IM "ollama app.exe" /F
taskkill /IM ollama.exe /F
set OLLAMA_NUM_PARALLEL=6 && setx OLLAMA_NUM_THREADS 6 && ollama serve
```

#### 2nd cmd window

**Note:** Wait 10-15 seconds after starting Ollama before running this command

- Run the command from the correct directory

```cmd
cd my-obsidian-wiki
set OLLAMA_NUM_PARALLEL=6 && set OLLAMA_NUM_THREADS=6 && olw run
```



### Output
- Drafts are saved to `wiki/` directory
- Review progress with: `olw review`
- Approve articles with: `olw review --approve all`

## Troubleshooting

**"Port 11434 already in use"?**
```cmd
taskkill /IM "ollama app.exe" /F
taskkill /IM ollama.exe /F
```

**"ModuleNotFoundError: No module named 'obsidian_llm_wiki'"?**
```cmd
pip install -e .
```

**Need a different LLM provider?**
```cmd
olw setup
```
Supports Groq, Together AI, LM Studio, vLLM, Azure OpenAI, etc.

## See Also
- `README.md` for full feature documentation
- `olw --help` for all commands

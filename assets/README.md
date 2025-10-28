# Assets

This directory contains interactive assets and tools for the GIFT framework.

## Structure

```
assets/
├── twitter_bot/           # Automated Twitter bot for GIFT content
│   ├── content_generator_en.py    # English content generator
│   ├── twitter_bot_v2.py          # Main bot script (API v2)
│   ├── scheduler.py               # Automated scheduler
│   ├── config.py                  # Bot configuration (API keys)
│   └── README.md                  # Bot documentation
│
└── visualizations/        # Interactive visualizations
    ├── e8_root_system_3d.ipynb    # E₈ root system 3D visualization
    ├── precision_dashboard.ipynb   # Precision comparison dashboard
    ├── dimensional_reduction_flow.ipynb  # Dimensional reduction animation
    └── outputs/                    # Generated figures and HTML
```

## Twitter Bot

The GIFT Twitter Bot automatically posts daily scientific content about the framework:

- **Daily posts**: 9:00 AM (random content)
- **Weekly summaries**: Monday 10:00 AM
- **Monthly highlights**: 1st of month 11:00 AM
- **Content types**: 8 different categories (precision, math, experimental, etc.)
- **Language**: English (international audience)

### Quick Start

```bash
cd assets/twitter_bot
pip install -r requirements.txt
python twitter_bot_v2.py
```

### Automated Scheduling

```bash
cd assets/twitter_bot
python scheduler.py
```

## Visualizations

Interactive Jupyter notebooks for exploring the GIFT framework:

- **E₈ Root System**: 3D visualization of the 240-root structure
- **Precision Dashboard**: All 34 observables vs experimental data
- **Dimensional Reduction**: Animation of E₈×E₈ → AdS₄×K₇ → SM

### Running Visualizations

```bash
cd assets/visualizations
jupyter notebook
```

Or use online platforms:
- [Binder](https://mybinder.org/v2/gh/gift-framework/GIFT/main?filepath=assets/visualizations/)
- [Google Colab](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/)

## Documentation

- **Twitter Bot**: See `assets/twitter_bot/README.md` for complete setup instructions
- **Visualizations**: See `assets/visualizations/README.md` for detailed usage
- **Main Framework**: See root `README.md` for theoretical background

## Contributing

Contributions to assets are welcome! Please see `CONTRIBUTING.md` for guidelines.

### Twitter Bot Contributions
- New content templates
- Improved scheduling options
- Additional language support
- Enhanced error handling

### Visualization Contributions
- New interactive plots
- Additional parameter exploration
- Performance optimizations
- Export format improvements

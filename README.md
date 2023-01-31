# Pixeltable

Pixeltable presents a dataframe-like interface to image and video data.

## Installation

1. Install Postgres

    On MacOS, [postgresapp.com](postgresapp.com) is a convenient way to do that.

2. `pip install pixeltable`

3. Install additional dependencies
   - Install PyTorch (required for CLIP): see [here](https://pytorch.org/get-started/locally/).
   - Install CLIP from [here](https://github.com/openai/CLIP).
   - If you want to work with videos, also install `ffmpeg`.

## Setup

Pixeltable requires a home directory and a Postgres database, both are created automatically the first time you create a Pixeltable client (see below).
The location of the home directory is `~/.pixeltable` (or the value of the `PIXELTABLE_HOME` environment variable);
the name of the Postgres database is `pixeltable` (or the value of the `PIXELTABLE_DB` environment variable).

## Overview

Import convention:
```
import pixeltable as pt
```

### Create a client
```
cl = pt.Client()
```

### Create a database
```
db = cl.create_db('db1')
```

### Create a table with video data
```
c1 = pt.Column('video', pt.VideoType())
c2 = pt.Column('frame_idx', pt.IntType())
c3 = pt.Column('frame', pt.ImageType())
t = db.create_table(
    'video_table', [c1, c2, c3],
    extract_frames_from='video',
    extracted_frame_col='frame',
    extracted_frame_idx_col='frame_idx',
    extracted_fps=1)
```

### Query table

|Action|Code|
|----|----|
| Look at 10 rows | `t.show(10)` |
| Look at all rows | `t.show(0)` |
| Look at row for frame 15 | `t[t.frame_idx == 15].show()` |
| Look at all frames before index 15 | `t[t.frame_idx < 15][t.frame].show(0)` |

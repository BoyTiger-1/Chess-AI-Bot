import React from 'react';
import { Box, Button, IconButton, InputAdornment, TextField, Tooltip } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import BookmarkAddIcon from '@mui/icons-material/BookmarkAdd';
import { useTranslation } from 'react-i18next';
import { useSearch } from '@/app/state/search';

export function GlobalSearch() {
  const { t } = useTranslation();
  const { query, setQuery, saveFilter } = useSearch();

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <TextField
        fullWidth
        size="small"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={t('common.search')}
        inputProps={{ 'aria-label': t('common.search') }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          )
        }}
      />
      <Tooltip title="Save filter">
        <IconButton
          aria-label="Save current search as filter"
          onClick={() => {
            const finalName = (query || 'Filter').slice(0, 40);
            saveFilter(finalName, query);
          }}
          disabled={!query.trim()}
        >
          <BookmarkAddIcon />
        </IconButton>
      </Tooltip>
      <Button
        variant="outlined"
        size="small"
        sx={{ display: { xs: 'none', md: 'inline-flex' } }}
        onClick={() => setQuery('')}
        disabled={!query}
      >
        Clear
      </Button>
    </Box>
  );
}

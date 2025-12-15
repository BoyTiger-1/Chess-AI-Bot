import React from 'react';
import { Button } from '@mui/material';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { useTranslation } from 'react-i18next';

export function ExportPdfButton({
  targetRef,
  fileName
}: {
  targetRef: React.RefObject<HTMLElement>;
  fileName: string;
}) {
  const { t } = useTranslation();
  const [busy, setBusy] = React.useState(false);

  return (
    <Button
      variant="outlined"
      size="small"
      startIcon={<PictureAsPdfIcon />}
      disabled={busy}
      onClick={async () => {
        if (!targetRef.current) return;
        setBusy(true);
        try {
          const canvas = await html2canvas(targetRef.current, {
            backgroundColor: null,
            scale: Math.min(2, window.devicePixelRatio || 1)
          });
          const imgData = canvas.toDataURL('image/png');

          const pdf = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4' });
          const pageWidth = pdf.internal.pageSize.getWidth();
          const pageHeight = pdf.internal.pageSize.getHeight();

          const ratio = Math.min(pageWidth / canvas.width, pageHeight / canvas.height);
          const w = canvas.width * ratio;
          const h = canvas.height * ratio;

          pdf.addImage(imgData, 'PNG', (pageWidth - w) / 2, (pageHeight - h) / 2, w, h);
          pdf.save(fileName);
        } finally {
          setBusy(false);
        }
      }}
    >
      {t('common.exportPdf')}
    </Button>
  );
}

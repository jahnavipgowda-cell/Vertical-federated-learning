export default function PageHeader({ eyebrow, title, description, right }) {
  return (
    <div className="flex items-start justify-between gap-6 mb-8">
      <div>
        {eyebrow && <div className="label-eyebrow mb-2">{eyebrow}</div>}
        <h1 className="text-3xl font-display font-semibold text-textPrimary">{title}</h1>
        {description && <p className="text-textMuted mt-2 max-w-2xl leading-relaxed">{description}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}
